# GPT-2 Instruction Fine-Tuning (Alpaca Dataset)

This project demonstrates how to fine-tune **GPT-2** for instruction-following tasks using the **Stanford Alpaca dataset** and Hugging Face Transformers.

The model is trained to generate responses based on structured prompts containing instructions and optional inputs.

---

## Overview

- **Base Model:** `openai-community/gpt2`
- **Fine-Tuning Type:** Full Model Fine-Tuning
- **Dataset:** `tatsu-lab/alpaca`
- **Framework:** PyTorch + Hugging Face Transformers

---

## What This Project Does

- Converts Alpaca dataset into instruction-style prompts  
- Applies tokenization with proper padding & truncation  
- Uses **loss masking** so the model only learns to generate responses  
- Fine-tunes GPT-2 using Hugging Face `Trainer`  

## Dataset Formatting

Each example from the dataset is converted into a structured instruction format that GPT-2 can understand:

```text
Below is an instruction that describes a task...

### Instruction:
<instruction>

### Input:
<input>

### Response:
<output>
```

## 🔤 Tokenization Details

The dataset is tokenized using the GPT-2 tokenizer with the following configuration:

- **Max Length:** 512 tokens  
- **Padding:** `max_length`  
- **Truncation:** Enabled  

This ensures all sequences are of uniform length, making batching efficient during training.

---

## Loss Masking Strategy

To properly train the model for instruction-following tasks, we apply **selective loss masking**:

- Tokens before `"### Response:"` → masked (`-100`)  
- Tokens after → used for training  

### Why this is important?

- Prevents the model from simply copying the prompt  
- Forces the model to **learn meaningful response generation**  
- Improves instruction-following capability  

The model **only learns from the response**, not the instruction.

## 🧪 Training Details

| Parameter             | Value                        |
|----------------------|------------------------------|
| Batch Size           | 2                            |
| Epochs               | 3                            |
| Max Sequence Length  | 512                          |
| Precision            | FP16 (if CUDA available)     |

---

## Training Pipeline

1. Load pretrained GPT-2  
2. Format dataset into instruction prompts  
3. Tokenize inputs  
4. Apply loss masking  
5. Train using Hugging Face `Trainer`

## Evaluation / Inference

The `eval.py` script loads the trained GPT-2 model (with adapters) and generates responses for new instructions.

## Results


<img width="1270" height="316" alt="image" src="https://github.com/user-attachments/assets/d62ade07-4642-46be-865d-eb4e3faed25d" />


<img width="1270" height="316" alt="image" src="https://github.com/user-attachments/assets/65d413b0-8d9d-4f82-aed6-ba41e3dcc4c7" />


<img width="1312" height="353" alt="image" src="https://github.com/user-attachments/assets/505eb5a7-efee-4601-a7df-6400c745610e" />





