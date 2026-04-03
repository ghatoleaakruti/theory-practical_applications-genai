# Theory & Practical Applications of Generative AI
### CSYE 7380 — Northeastern University (Spring 2026)

Coursework and projects exploring large language models, prompt engineering, RAG architectures, agentic systems, and LLM fine-tuning.

---

## 🏆 Midterm Project — Mistral-7B Text-to-SQL Fine-Tuning

**Fine-tuning Mistral-7B-Instruct-v0.2 for Text-to-SQL generation using QLoRA**

### What It Does
Takes a natural language question + database schema → generates correct SQL query with explanation.

### Approach
- **Base model:** Mistral-7B-Instruct-v0.2
- **Dataset:** gretelai/synthetic_text_to_sql (1,000 training samples)
- **Method:** QLoRA (4-bit NF4 quantization + LoRA, 0.36% trainable parameters)
- **Training:** 1 epoch on Google Colab T4 GPU (free tier)
- **Framework:** TRL SFTTrainer + HuggingFace Transformers
- **Interface:** Gradio web UI

### Results
| Metric | Before Fine-Tuning | After Fine-Tuning |
|---|---|---|
| Training Loss | 1.17 | 0.36 |
| Exact Match (20 samples) | 5% | 20% |
| Output Quality | Wrong SQL, messy format | Correct SQL, clean format |

### Key Files
- `MidTerm.ipynb` — Main fine-tuning notebook
- `Midterm_PartB.ipynb` — Evaluation and Gradio interface
- `Fine-Tuning-Mistral-7B-for-Text-to-SQL-Using-QLoRA.pdf` — Full report
- `MidTerm_PartA.pptx` — Project presentation Part A
- `MidTerm_PartB.pptx` — Project presentation Part B

---

## 📚 Assignments

| Assignment | Topic |
|---|---|
| Assignment 1 | LLM fundamentals and prompt engineering |
| Assignment 2 | RAG architecture and vector databases |
| Assignment 3 | Agent-based systems |
| Assignment 4 | LLM evaluation techniques |
| Assignment 5 | Practical GenAI system design |
| Assignment 6 | Advanced LLM applications |
| Assignment 7 | Agentic workflows |

---

## 🔧 Tech Stack

- **Models:** Mistral-7B · HuggingFace Transformers
- **Fine-tuning:** QLoRA · PEFT · TRL · BitsAndBytes
- **Interface:** Gradio
- **Environment:** Google Colab · Python
- **Libraries:** PyTorch · Pandas · NumPy

---

## 🚀 Final Project

*Coming May 2026 — CAD Document Intelligence System combining rule-based validation and LLM reasoning to detect inconsistencies in engineering drawings.*

---

*Part of MS Information Systems curriculum at Northeastern University*
