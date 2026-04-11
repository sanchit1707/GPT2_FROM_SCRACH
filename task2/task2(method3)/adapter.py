import torch
from transformers import GPT2LMHeadModel
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers import Trainer,TrainingArguments

from datasets import load_dataset

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

class Adapter(nn.Module):
    def __init__(self,hidden_size,bottleneck=64):
        super().__init__()

        self.lin1=nn.Linear(hidden_size, bottleneck)
        self.lin2=nn.Linear(bottleneck,hidden_size)
        nn.init.normal_(self.lin1.weight, std=1e-3)
        nn.init.zeros_(self.lin1.bias)

        nn.init.zeros_(self.lin2.weight)   
        nn.init.zeros_(self.lin2.bias)

    def forward(self,x):
        return x+ self.lin2(F.gelu(self.lin1(x)))
    
def addadapter_to_transformer(model, bottleneck=64):
    hidden_size = model.config.n_embd

    for block in model.transformer.h:
        block.adapter1 = Adapter(hidden_size, bottleneck)
        block.adapter2 = Adapter(hidden_size, bottleneck)

        old_forward = block.forward

        def make_forward(old_forward):
            def new_forward(self, *args, **kwargs):

                outputs = old_forward(*args, **kwargs)

                if isinstance(outputs,tuple):

                    hidden_states = outputs[0]

                    hidden_states = self.adapter1(hidden_states)
                    hidden_states = self.adapter2(hidden_states)

                    return (hidden_states,1)+outputs[1:]
                
                else:
                    hidden_states=outputs
                    hidden_states=self.adapter1(hidden_states)
                    hidden_states=self.adapter2(hidden_states)

                    return hidden_states

            return new_forward
        
        block.forward=make_forward(old_forward).__get__(block,block.__class__)

    return model

model = addadapter_to_transformer(model)
model.config.use_cache = False
model.to(device)

for name, param in model.named_parameters():
    if "adapter" not in name:
        param.requires_grad = False

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
    output_dir="./gpt2-alpaca-adapter",
    per_device_train_batch_size=4,  
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=100,
    fp16=torch.cuda.is_available()
)
trainer = Trainer(model=model,args=training_params,train_dataset=tokenized_ds["train"])


trainer.train()
model.save_pretrained("./gpt2-alpaca-adapter")
tokenizer.save_pretrained("./gpt2-alpaca-adapter")


