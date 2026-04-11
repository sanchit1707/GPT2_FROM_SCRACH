import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


base_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")


model = PeftModel.from_pretrained(base_model, "./gpt2-alpaca-lora")

model.to(device)
model.eval()


tokenizer = AutoTokenizer.from_pretrained("./gpt2-alpaca-lora")

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

prompt = """Below is an instruction that describes a task.

### Instruction:
Explain neural networks in simple terms.

### Response:
"""

print(generate_response(prompt))
