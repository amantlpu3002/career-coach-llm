# Career Coach LLM

Fine-tuned TinyLlama-1.1B for career coaching using QLoRA (4-bit NF4).
Trained on RTX 3050 consumer GPU.

## Results
| Metric   | Base Model | Fine-tuned | Improvement |
|----------|-----------|------------|-------------|
| ROUGE-L  | 0.18      | X.XX       | +XX%        |

## Tech Stack
- Model: TinyLlama-1.1B-Chat-v1.0
- Method: QLoRA (PEFT, bitsandbytes 4-bit NF4)
- LoRA rank: r=8, alpha=16
- Hardware: NVIDIA RTX 3050
- Demo: Flask web app

## Run
```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

## Resume Bullet
Career Coach LLM | Python · QLoRA · TinyLlama · Hugging Face
- Fine-tuned TinyLlama-1.1B on career-coaching dataset using QLoRA on RTX 3050
- Improved ROUGE-L by XX% over base model
- Deployed as live web demo
