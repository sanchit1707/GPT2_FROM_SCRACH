import torch 
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers import Trainer,TrainingArguments
from datasets import load_dataset

from peft import LoraConfig, get_peft_model, TaskType
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lora_config=LoraConfig(r=8,
                       lora_alpha=16,
                       target_modules=["c_attn"],
                       lora_dropout=0.1,
                       task_type=TaskType.CAUSAL_LM)
model=AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
model=get_peft_model(model,lora_config)
model.to(device)

tokenizer=AutoTokenizer.from_pretrained("openai-community/gpt2")

ds=load_dataset("tatsu-lab/alpaca")

def format_example(example):
    if example["input"]:
        prompt = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    else:
        prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n{example['output']}"
        )
    return {"text": prompt}

ds=ds.map(format_example)

tokenizer.pad_token=tokenizer.eos_token
model.config.pad_token_id=tokenizer.pad_token_id

def tokenize_function(example):
    full_text = example["text"]
    
    tokenized = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=512
    )

    labels = tokenized["input_ids"].copy()

    # Find where response starts
    response_start = full_text.find("### Response:\n")
    if response_start == -1:
        response_start = 0
    else:
        response_start += len("### Response:\n")

    # Tokenize only the prefix to know cutoff
    prefix_tokens = tokenizer(
        full_text[:response_start],
        truncation=True,
        max_length=512
    )["input_ids"]

    cutoff = len(prefix_tokens)

    # Mask everything before response
    labels[:cutoff] = [-100] * cutoff

    tokenized["labels"] = labels
    return tokenized

tokenized_ds = ds.map(tokenize_function, batched=False)
tokenized_ds = tokenized_ds.remove_columns(
    ["instruction", "input", "output", "text"])

training_params = TrainingArguments(
    output_dir="./gpt2-alpaca-lora",
    per_device_train_batch_size=4,  
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    fp16=torch.cuda.is_available()
)
trainer = Trainer(model=model,args=training_params,train_dataset=tokenized_ds["train"])

trainer.train()
model.save_pretrained("./gpt2-alpaca-lora")
tokenizer.save_pretrained("./gpt2-alpaca-lora")

