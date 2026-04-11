# GPT-2 From Scratch (PyTorch)

A minimal implementation of a GPT-2 style language model built from scratch using PyTorch, along with a training pipeline on the Shakespeare dataset.

---

## Overview

This project implements a **decoder-only Transformer** trained for **next-token prediction**. The model learns to generate text by predicting the next token given previous context.

---

## Architecture

### Components

- **Token Embedding** – Converts token IDs into dense vectors  
- **Positional Embedding** – Adds order information to tokens  
- **Masked Multi-Head Self-Attention** – Prevents attending to future tokens (causal masking)  
- **Feedforward Network (MLP)** – Expands and compresses representation using GELU activation  
- **LayerNorm + Residual Connections** – Stabilizes training  
- **Output Projection** – Maps embeddings to vocabulary logits (weight tied with embeddings)

### Transformer Block
Input
→ LayerNorm → Masked Attention → Residual
→ LayerNorm → Feedforward → Residual

## Dataset & Tokenization

- Dataset: `shakespeare.txt`  
- Tokenizer: `tiktoken` (GPT-2 encoding)

```python
enc = tiktoken.get_encoding("gpt2")
tokens = torch.tensor(enc.encode(text), dtype=torch.long)
decoded = enc.decode(tokens.tolist())
assert decoded == text
```

## Batch Sampling

### Why random sampling?
Breaks sequential correlation in the dataset
Improves generalization across different parts of data
Makes training stochastic (like SGD)
Prevents overfitting to specific text order

## Training Objective

The model is trained using **next-token prediction**:

Loss function:

```python
loss_fn = torch.nn.CrossEntropyLoss()
```

## training loop

Details
-Optimizer: AdamW (lr = 3e-4)
-Periodic checkpoint saving
-Loss is printed during training

## Text Generation

### Core idea

```python
probs = torch.softmax(logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

### Why sampling instead of argmax?
-Produces more diverse and natural text
-Avoids repetitive outputs

## further improvments
-Add dropout
-Add dropout

## purpose 
getting into cynaptics(hopefully)

## How to Run (Evaluation / Inference)

### ⚙️ Steps to Run

1. **Train the model (or use an existing checkpoint)**  
   Ensure you have a trained model checkpoint stored in:
   `checkpoints/`
   
2. **Update checkpoint path**  
Replace:
```python
r'S:\gpt2_from_scrach\checkpoints\checkpoint_38500.pth'
```

3. **run the `eval.py` script**

