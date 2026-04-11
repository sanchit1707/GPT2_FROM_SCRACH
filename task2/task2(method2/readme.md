# GPT-2 LoRA Fine-Tuning on Alpaca Dataset

## Overview

This project fine-tunes **GPT-2** using **LoRA (Low-Rank Adaptation)** on the **Alpaca dataset** for instruction-following tasks.
The training setup is optimized to:

* Reduce memory usage using LoRA
* Train efficiently on limited GPU resources
* Focus learning only on response generation (not prompt)

---

## Features

*  LoRA-based parameter-efficient fine-tuning
*  Instruction + input → response formatting
*  Label masking (only response contributes to loss)
*  Compatible with Hugging Face `Trainer`
*  Works on GPU/CPU

---

## Model Details

* Base Model: `gpt2`
* Fine-tuning Method: LoRA
* Task Type: Causal Language Modeling

### LoRA Configuration

```python
r = 8
lora_alpha = 16
target_modules = ["c_attn"]
lora_dropout = 0.1
```

---

## Dataset

* Dataset: `tatsu-lab/alpaca`
* Format:

  * `instruction`
  * `input`
  * `output`

---

## Training Pipeline

### 1. Prompt Formatting

Each example is converted into:

```
Below is an instruction...

### Instruction:
...

### Input:
...

### Response:
...
```

---

### 2. Tokenization

* Max length: `512`
* Padding: `max_length`
* Truncation: enabled

---

### 3. Label Masking 

Only the **response portion** is used for loss computation.

* Tokens before `"### Response:\n"` → masked with `-100`
* Padding tokens → masked with `-100`

---

### 4. Training Configuration

```python
per_device_train_batch_size = 4
num_train_epochs = 3
logging_steps = 10
save_steps = 500
fp16 = True (if GPU available)
```

---

## How to Run

### 1. Install Dependencies

```bash
pip install torch transformers datasets peft
```

### 2. Run Training

```bash
python train.py
```

---

## Output

The trained model and tokenizer are saved to:

```
./gpt2-alpaca-lora
```

---

## How to Load the Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
model = PeftModel.from_pretrained(base_model, "./gpt2-alpaca-lora")

tokenizer = AutoTokenizer.from_pretrained("./gpt2-alpaca-lora")
```

---

## Inference Example

```python
prompt = """Below is an instruction...

### Instruction:
Explain gravity

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## results

these are the results i got after training

![WhatsApp Image 2026-04-11 at 22 10 04](https://github.com/user-attachments/assets/3bf40494-e927-4a94-9641-9791dbeeee43)

![WhatsApp Image 2026-04-11 at 22 11 39](https://github.com/user-attachments/assets/a085d780-f8f0-4d72-9fa5-a2c96ae9bb0e)

![WhatsApp Image 2026-04-11 at 22 13 25](https://github.com/user-attachments/assets/15aa91f6-8f4d-4b88-9adb-81c3d8a10118)



