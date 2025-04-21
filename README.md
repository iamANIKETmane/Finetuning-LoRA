# 🧠 LoRA-RoBERTa: Parameter-Efficient Text Classification on AGNews

This repository contains our Deep Learning mini-project at NYU, where we fine-tuned a RoBERTa model on the AGNews text classification dataset using **Low-Rank Adaptation (LoRA)** under a constraint of 1 million trainable parameters. Our final model achieves a test accuracy of **92.96%** with just **796,420 trainable parameters**—demonstrating the power of parameter-efficient tuning.

---

## 📌 Project Members

- **Aniket Mane** — am14661@nyu.edu  
- **Subhan Akhtar** — sa8580@nyu.edu  
- **Pranav Motarwar** — pm3891@nyu.edu

---

## 🔍 Overview

Traditional fine-tuning of large language models is computationally expensive and resource-intensive. **LoRA** (Low-Rank Adaptation) introduces a highly efficient alternative by injecting low-rank trainable matrices into frozen pre-trained weights, dramatically reducing the number of trainable parameters while retaining performance.

In this project, we:
- Used `roberta-base` as the frozen backbone model.
- Injected LoRA modules into selected layers: **[0, 1, 5, 10, 11]**.
- Tuned LoRA hyperparameters: **rank `r = 12`**, **scaling factor `α = 32`**, and **dropout rate `0.06`**.
- Employed **cosine learning rate decay** with a warm-up ratio of `0.15`.
- Trained the model over **6 epochs**, using **AdamW optimizer** and **gradient accumulation**.

---

## 🛠️ Setup Instructions

```bash
# Clone the repo
git clone https://github.com/YOUR-TEAM/LoRA-AGNews.git
cd LoRA-AGNews

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
