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


## 📓 Notebook

The full training and evaluation pipeline is documented in the Jupyter notebook:

📄 [`LoRA_DLProject-2.ipynb`](./LoRA_DLProject-2.ipynb)

---

## 📊 Results

| Metric             | Value                 |
|--------------------|-----------------------|
| **Test Accuracy**  | 92.96%                |
| **Trainable Params** | 796,420 (~0.63%)    |
| **Base Model**     | `roberta-base` (frozen) |

> ✅ These results were obtained by tuning only the LoRA modules—**not** the full model weights.

---

## 🔬 Methodology Summary

**Base Model**: [`roberta-base`](https://huggingface.co/roberta-base) from HuggingFace (frozen)

**LoRA Injection Layers**: `[0, 1, 5, 10, 11]`

### LoRA Parameters
- **Rank (`r`)**: 12
- **Scaling Factor (`α`)**: 32
- **Dropout**: 0.06

### Training Configuration
- **Optimizer**: AdamW
- **Learning Rate**: `3e-4`
- **Scheduler**: CosineDecay with warmup (`warmup_ratio = 0.15`)
- **Batch Size**: 16 (train), 32 (eval)
- **Epochs**: 6
- **Gradient Accumulation Steps**: 2

---

## 💡 Key Insights

- **Layer Selection**: Injecting LoRA into deeper layers (10, 11) produced better performance.
- **Regularization**: Adding custom dropout (0.06) helped with generalization.
- **Hyperparameter Tuning**: The choice of `α` and `r` directly impacted stability and accuracy.
- **Efficiency**: LoRA achieved strong performance using less than 1% of the model's parameters.

---

## 📚 References

- Hu et al. (2021): [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685.pdf)
- [HuggingFace PEFT Library](https://github.com/huggingface/peft)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [AGNews Dataset (Kaggle)](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
- ChatGPT, OpenAI – Assisted in explanation, structure, and formatting



---

## 🤝 Acknowledgements

This project was completed as part of the **Deep Learning course at NYU Tandon School of Engineering** in **Spring 2025**.

We would like to thank our professor and teaching assistants for their support and guidance throughout the project.

---
