# GPT-2 Fine-Tuning with Adapters (Alpaca Dataset)

This project demonstrates how to **fine-tune GPT-2 efficiently using lightweight Adapter modules** on the Stanford Alpaca dataset.

Instead of updating all model parameters, we inject **trainable adapters into transformer blocks**, drastically reducing compute and memory usage.

---

## Overview

- Base Model: `openai-community/gpt2`
- Fine-tuning Method: **Adapter-based tuning**
- Dataset: `tatsu-lab/alpaca`
- Framework: PyTorch + Hugging Face Transformers

---

## Idea

Rather than full fine-tuning:
-Freeze original GPT-2 weights  
-Insert small **Adapter layers** inside transformer blocks  
-Train only adapter parameters  

This makes training:
- Faster 
- Memory efficient 

## Architecture

### Adapter Module

Each transformer block gets **two adapters**:
Input → Transformer Block → Adapter1 → Adapter2 → Output

Adapter structure:

- Down projection (hidden → bottleneck)
- GELU activation
- Up projection (bottleneck → hidden)
- Residual connection

---

## Setup Instructions

### 1. Install Dependencies

```bash
pip install torch transformers datasets
```

### 2.dataset formatting
Below is an instruction that describes a task...
```
### Instruction:
...

### Input:
...

### Response:
...
```
## 🧪 Training Details

| Parameter             | Value                          |
|----------------------|--------------------------------|
| Batch Size           | 4                              |
| Epochs               | 3                              |
| Learning Rate        | 1e-4                           |
| Max Sequence Length  | 512                            |
| Optimizer            | AdamW (default Trainer)        |
| Precision            | FP16 (if CUDA available)       |

## Loss Masking Strategy

Only the **response part** contributes to the loss.

- Tokens before `"### Response:"` → masked (`-100`)
- Tokens after → used for training  

This ensures the model learns **generation**, not prompt copying.

---

## Adapter 

Adapters are dynamically added to each transformer block by overriding the forward pass:

```python
block.adapter1 = Adapter(...)
block.adapter2 = Adapter(...)
```
## Parameter Freezing

```python
for name, param in model.named_parameters():
    if "adapter" not in name:
        param.requires_grad = False
```
## Evaluation / Inference

`eval.py` loads the trained adapter-based GPT-2 model and generates responses for new instructions.

---

### Loading the Trained Model

The model is initialized with base GPT-2 weights and then adapter weights are loaded from a `.safetensors` file:

```python
from safetensors.torch import load_file

state_dict = load_file("path/to/model.safetensors")
model.load_state_dict(state_dict, strict=False)
```

## Results

these are the results i got 
i would have uploaded the weights file but gut hub do not allow (i am sed abot that)

<img width="1242" height="258" alt="image" src="https://github.com/user-attachments/assets/4d95ad52-6b85-4358-adba-2ab051e6e4d5" />

<img width="1278" height="293" alt="image" src="https://github.com/user-attachments/assets/7f8ef174-dabd-4ed2-96c9-e418cce754ce" />

<img width="1134" height="379" alt="image" src="https://github.com/user-attachments/assets/7ea488a4-743a-41b5-9a7a-ed9b375dd6dc" />



