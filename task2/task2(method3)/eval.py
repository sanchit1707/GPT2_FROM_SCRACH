import torch
from transformers import GPT2LMHeadModel
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from safetensors.torch import load_file


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
state_dict = load_file(
    r"S:\gpt2_from_scrach\task2\task2(method3)\gpt2-alpaca-adapter\model.safetensors"
)

model.load_state_dict(state_dict, strict=False)
model.config.use_cache = False
model.to(device)


for name, param in model.named_parameters():
    if "adapter" not in name:
        param.requires_grad = False

tokenizer=AutoTokenizer.from_pretrained("openai-community/gpt2")






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


def generate_response(instruction, input_text=""):
    if input_text:
        prompt = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
        )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    print("==== TEST 1 ====")
    print(generate_response("Explain gravity in simple terms"))

    print("\n==== TEST 2 ====")
    print(generate_response(
        "Translate to French",
        "Hello, how are you?"
    ))

    print("\n==== TEST 3 ====")
    print(generate_response("Write a short poem about AI"))

