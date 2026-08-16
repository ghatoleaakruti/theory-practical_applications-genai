# Generative AI & Distributed Inference Systems

Deep learning systems, LLM applications, and production ML pipelines. Coursework + projects from CSYE 7380 (Spring 2026, Northeastern University).

---

## 🚀 Main Project — DDPM Inference Pipeline

**Distributed Generative AI System with Async Processing**

A production-grade DDPM inference engine with Kafka pub/sub orchestration, Redis caching, and 13× speedup via distillation research.

### What It Does
- Train DDPM from scratch (PyTorch): 62.7M-parameter U-Net on CIFAR-10
- Serve inference via async Kafka pipeline (Request → Inference → Result agents)
- Cache repeated requests in Redis, store results in S3
- Measure distillation speedups (SDXL-Turbo, LCM, DMD2)

### Architecture
```
FastAPI → Kafka (3 agents) → Redis Cache → PostgreSQL History → S3 Storage
         ↓
      Docker + CI/CD
```

### Key Results
- **Model**: Loss convergence 1.0 → 0.05 on CIFAR-10
- **Speedup**: Surveyed 6 one-step distillation papers, achieved **13× inference speedup**
- **Pipeline**: Async processing without blocking API responses
- **Scale**: Dockerized, CI/CD ready, production patterns

### Tech Stack
**ML**: PyTorch, HuggingFace, SDXL, LCM  
**Infrastructure**: Kafka, Redis, PostgreSQL, S3, Docker  
**API**: FastAPI, async/await  
**Tools**: GitHub Actions, CloudWatch  

### Files
- `ddpm_training.ipynb` — Full training pipeline, loss tracking, CIFAR-10
- `kafka_async_pipeline.py` — 3-agent Kafka orchestration
- `distillation_research.md` — Survey of one-step methods + speedup measurements
- `docker-compose.yml` — Full local setup

---

## 📚 Coursework & Supporting Notebooks

**CSYE 7380: Theory & Practical Applications of Generative AI**

### Assignments (Learning Foundations)
Hands-on exploration of LLM fundamentals, RAG, agents, and evaluation:

| Assignment | Topic | Notebook |
|------------|-------|----------|
| 1 | LLM fundamentals & prompt engineering | `Assignment1.ipynb` |
| 2 | RAG architecture & vector databases | `Assignment2.ipynb` |
| 3 | Agent-based systems | `Assignment3.ipynb` |
| 4 | LLM evaluation techniques | `Assignment4.ipynb` |
| 5 | Practical GenAI system design | `Assignment5.ipynb` |
| 6 | Advanced LLM applications | `Assignment6.ipynb` |
| 7 | Agentic workflows | `Assignment7.ipynb` |

### Midterm Project — Mistral-7B Text-to-SQL Fine-Tuning
Fine-tuned Mistral-7B-Instruct for SQL generation using QLoRA (0.36% trainable parameters).

**Results**: Training loss 1.17 → 0.36, exact match 5% → 20%

Files: `MidTerm.ipynb`, `Midterm_PartB.ipynb`, reports and presentations

---

## 🔧 Tech Stack

**Models**: PyTorch · HuggingFace Transformers · Mistral-7B · SDXL  
**Distributed Systems**: Kafka · Redis · PostgreSQL · S3  
**Infrastructure**: Docker · GitHub Actions · FastAPI  
**ML Frameworks**: PyTorch · QLoRA · PEFT · TRL  
**Tools**: Gradio · Jupyter · Google Colab  



---

**Copy this in and push. Done.** 💪
