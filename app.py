"""
CareerMind — Flask Backend
Fine-tuned TinyLlama-1.1B with QLoRA adapter

Endpoints:
  POST /api/analyze   → scores + keyword gap
  POST /api/rewrite   → AI summary rewriter
  POST /api/interview → tailored interview questions
  GET  /api/health    → model status check
"""

import os, re, json, time, logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static")
CORS(app)

# ─── Model globals ────────────────────────────────────────────────────────────
model = None
tokenizer = None
MODEL_LOADED = False
ADAPTER_PATH = os.environ.get(
    "ADAPTER_PATH",
    r"D:/career-coach-notebooks/notebooks/outputs/career-coach-qlora/final-adapter"
)
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ─── Load model ───────────────────────────────────────────────────────────────
def load_model():
    global model, tokenizer, MODEL_LOADED
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        log.info("Loading tokenizer from base model…")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        tokenizer.pad_token = tokenizer.eos_token

        use_gpu = torch.cuda.is_available()
        log.info(f"CUDA available: {use_gpu}")

        if use_gpu:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                quantization_config=bnb_cfg,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            # CPU fallback — slower but works on HF Spaces free tier
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
            )

        if os.path.exists(ADAPTER_PATH):
            log.info(f"Loading LoRA adapter from {ADAPTER_PATH}…")
            model = PeftModel.from_pretrained(base, ADAPTER_PATH)
            log.info("✓ Fine-tuned adapter loaded")
        else:
            log.warning(f"Adapter not found at {ADAPTER_PATH} — using base model")
            model = base

        model.eval()
        MODEL_LOADED = True
        log.info("✓ Model ready")

    except Exception as e:
        log.error(f"Model load failed: {e}")
        MODEL_LOADED = False


def generate(prompt: str, max_new_tokens: int = 512) -> str:
    """Run inference; returns generated text."""
    import torch
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the new tokens
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ─── Prompt builders ──────────────────────────────────────────────────────────
def build_score_prompt(resume: str, job_desc: str = "") -> str:
    jd_section = f"\n\nJob Description:\n{job_desc}" if job_desc else ""
    return f"""<|system|>
You are CareerMind, an expert career coach. Analyze the resume and return ONLY a valid JSON object — no extra text.
</s>
<|user|>
Score this resume across 5 dimensions (0-100). Return JSON exactly like:
{{"overall":75,"impact":82,"keywords":68,"clarity":79,"ats":61,"summary":"One sentence overall assessment.","improvements":["improvement 1","improvement 2","improvement 3"]}}

Resume:
{resume}{jd_section}
</s>
<|assistant|>"""


def build_keyword_prompt(resume: str, job_desc: str) -> str:
    return f"""<|system|>
You are CareerMind, an ATS keyword specialist. Return ONLY valid JSON — no extra text.
</s>
<|user|>
Compare the resume to the job description. Return JSON exactly like:
{{"found":["Python","REST API","Git"],"missing":["Docker","CI/CD","AWS"],"suggested":["System Design","Unit Testing"]}}

Resume:
{resume}

Job Description:
{job_desc}
</s>
<|assistant|>"""


def build_rewrite_prompt(resume_summary: str, job_desc: str = "") -> str:
    jd_section = f"\nTarget role context: {job_desc[:300]}" if job_desc else ""
    return f"""<|system|>
You are CareerMind, a professional resume writer. Rewrite the summary to be impact-first, results-driven, and ATS-optimized.
</s>
<|user|>
Rewrite this professional summary. Keep it 3-4 sentences. Use strong action verbs and quantify where possible.{jd_section}

Original summary:
{resume_summary}
</s>
<|assistant|>"""


def build_interview_prompt(resume: str, job_desc: str = "") -> str:
    jd_section = f"\nJob Description:\n{job_desc}" if job_desc else ""
    return f"""<|system|>
You are CareerMind, an interview coach. Generate tailored interview questions based on this resume. Return ONLY valid JSON — no extra text.
</s>
<|user|>
Generate 5 tailored interview questions. Return JSON exactly like:
{{"questions":[{{"num":1,"question":"Tell me about...","type":"Behavioural"}},{{"num":2,"question":"How did you...","type":"Technical"}}]}}

Resume:
{resume}{jd_section}
</s>
<|assistant|>"""


# ─── JSON extraction helper ───────────────────────────────────────────────────
def extract_json(text: str) -> dict | None:
    """Pull first JSON object from model output (handles trailing text)."""
    # Try direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass
    # Find JSON block
    m = re.search(r'\{[\s\S]+\}', text)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


def score_fallback(resume: str) -> dict:
    """Deterministic fallback scorer when model output can't be parsed."""
    words = resume.lower().split()
    kw = ["python","sql","api","docker","aws","git","agile","ci","cd","kubernetes",
          "javascript","typescript","java","react","flask","fastapi","postgres"]
    matched = sum(1 for w in kw if w in words)
    kw_score = min(100, int(matched / len(kw) * 100) + 30)
    length_score = min(100, max(40, len(words) * 2))
    overall = int((kw_score + length_score) / 2)
    return {
        "overall": overall,
        "impact": min(100, overall + 7),
        "keywords": kw_score,
        "clarity": min(100, length_score + 5),
        "ats": max(40, kw_score - 7),
        "summary": "Analysis based on keyword density. Add more role-specific terms to improve your score.",
        "improvements": [
            "Add quantified achievements (e.g. 'improved latency by 40%')",
            "Include more ATS keywords matching your target role",
            "Strengthen your professional summary with impact-first language"
        ]
    }


def keyword_fallback(resume: str, job_desc: str) -> dict:
    """Fallback keyword gap when model output can't be parsed."""
    common_kw = ["python","sql","api","rest","docker","aws","git","agile",
                 "typescript","javascript","kubernetes","ci/cd","testing",
                 "postgresql","react","flask","fastapi","linux","bash","redis"]
    resume_lower = resume.lower()
    jd_lower = job_desc.lower()
    found = [k for k in common_kw if k in resume_lower and k in jd_lower]
    missing = [k for k in common_kw if k not in resume_lower and k in jd_lower][:6]
    suggested = [k for k in common_kw if k not in resume_lower and k not in jd_lower][:4]
    return {"found": found, "missing": missing, "suggested": suggested}


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/health")
def health():
    import torch
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL_LOADED,
        "adapter_present": os.path.exists(ADAPTER_PATH),
        "cuda": __import__("torch").cuda.is_available() if MODEL_LOADED else False,
        "base_model": BASE_MODEL,
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Score resume + keyword gap in one call."""
    data = request.get_json(force=True)
    resume = (data.get("resume") or "").strip()
    job_desc = (data.get("job_desc") or "").strip()

    if not resume:
        return jsonify({"error": "resume text is required"}), 400

    t0 = time.time()

    # --- Scores ---
    if MODEL_LOADED:
        try:
            raw = generate(build_score_prompt(resume, job_desc), max_new_tokens=256)
            scores = extract_json(raw) or score_fallback(resume)
        except Exception as e:
            log.error(f"Score generation error: {e}")
            scores = score_fallback(resume)
    else:
        scores = score_fallback(resume)

    # Ensure all required keys exist
    for key in ("overall","impact","keywords","clarity","ats"):
        scores.setdefault(key, 70)
    scores.setdefault("summary", "Analysis complete.")
    scores.setdefault("improvements", [])

    # --- Keywords ---
    if job_desc:
        if MODEL_LOADED:
            try:
                raw_kw = generate(build_keyword_prompt(resume, job_desc), max_new_tokens=200)
                keywords = extract_json(raw_kw) or keyword_fallback(resume, job_desc)
            except Exception as e:
                log.error(f"Keyword generation error: {e}")
                keywords = keyword_fallback(resume, job_desc)
        else:
            keywords = keyword_fallback(resume, job_desc)
    else:
        keywords = {"found": [], "missing": [], "suggested": []}

    elapsed = round(time.time() - t0, 2)
    return jsonify({"scores": scores, "keywords": keywords, "elapsed": elapsed})


@app.route("/api/rewrite", methods=["POST"])
def rewrite():
    """Rewrite a resume summary."""
    data = request.get_json(force=True)
    summary = (data.get("summary") or data.get("resume") or "").strip()
    job_desc = (data.get("job_desc") or "").strip()

    if not summary:
        return jsonify({"error": "summary text is required"}), 400

    t0 = time.time()
    if MODEL_LOADED:
        try:
            rewritten = generate(build_rewrite_prompt(summary, job_desc), max_new_tokens=300)
        except Exception as e:
            log.error(f"Rewrite error: {e}")
            rewritten = "[Model error — please try again]"
    else:
        rewritten = (
            "Results-driven professional with proven expertise in backend engineering and "
            "cross-functional collaboration. Delivered scalable REST APIs and optimized system "
            "performance in Agile environments, consistently exceeding team velocity targets. "
            "Passionate about clean architecture and continuous improvement."
        )

    elapsed = round(time.time() - t0, 2)
    return jsonify({"original": summary, "rewritten": rewritten, "elapsed": elapsed})


@app.route("/api/interview", methods=["POST"])
def interview():
    """Generate tailored interview questions."""
    data = request.get_json(force=True)
    resume = (data.get("resume") or "").strip()
    job_desc = (data.get("job_desc") or "").strip()

    if not resume:
        return jsonify({"error": "resume text is required"}), 400

    t0 = time.time()
    fallback_questions = [
        {"num": 1, "question": "Walk me through a technical challenge you solved and the impact it had.", "type": "Behavioural"},
        {"num": 2, "question": "How do you approach designing a new backend system from scratch?", "type": "Technical"},
        {"num": 3, "question": "Describe a time you had to explain a complex technical decision to non-technical stakeholders.", "type": "Communication"},
        {"num": 4, "question": "What's a gap in your skill set and what's your plan to close it?", "type": "Growth Mindset"},
        {"num": 5, "question": "How do you prioritise when you have multiple urgent tasks competing for your attention?", "type": "Work Style"},
    ]

    if MODEL_LOADED:
        try:
            raw = generate(build_interview_prompt(resume, job_desc), max_new_tokens=400)
            parsed = extract_json(raw)
            questions = parsed.get("questions", fallback_questions) if parsed else fallback_questions
        except Exception as e:
            log.error(f"Interview generation error: {e}")
            questions = fallback_questions
    else:
        questions = fallback_questions

    elapsed = round(time.time() - t0, 2)
    return jsonify({"questions": questions, "elapsed": elapsed})


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_model()
    port = int(os.environ.get("PORT", 8080))
    log.info(f"Starting CareerMind on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
