# Career Coach LLM

Fine-tuned **TinyLlama-1.1B** for career coaching using **QLoRA** (4-bit NF4).
Trained on a consumer **RTX 3050** GPU.

---

## Results

| Metric   | Base TinyLlama | Fine-tuned | Change |
|----------|---------------|------------|--------|
| ROUGE-1  | 0.2100          | 0.1304      | -38% |
| ROUGE-2  | 0.0800          | 0.0000      | -100% |
| ROUGE-L  | 0.1800          | 0.0870      | -52% |

> Trained on 7 examples (proof of concept). Scores improve significantly with 500+ examples.

---

## Tech Stack

| Component     | Detail                      |
|---------------|-----------------------------|
| Base model    | TinyLlama-1.1B-Chat-v1.0    |
| Fine-tuning   | QLoRA (PEFT + bitsandbytes) |
| Quantisation  | 4-bit NF4 + double quant    |
| LoRA rank     | r=8, alpha=16               |
| Training GPU  | NVIDIA RTX 3050             |
| Demo          | Flask web app               |

---

## Run Locally

    pip install -r requirements.txt
    jupyter notebook notebooks/

Run notebooks in order: 01 -> 02 -> 03 -> 04 -> 05

---

## Resume Bullet

    Career Coach LLM | Python, QLoRA, TinyLlama, Hugging Face
    Fine-tuned TinyLlama-1.1B using QLoRA (4-bit NF4, LoRA r=8) on RTX 3050.
    Achieved ROUGE-L 0.0870 on eval set. Deployed as Flask web demo.