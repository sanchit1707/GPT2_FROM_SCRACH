import torch 
import torch.nn as nn
import torch.nn.functional as f
from GPT2 import GPT2
import params as pa
import tiktoken
from train import generate


model=GPT2()

model.load_state_dict(torch.load (r'S:\gpt2_from_scrach\checkpoints\checkpoint_38500.pth',map_location=pa.device,weights_only=True))
model.to(pa.device)
model.eval()

enc=tiktoken.get_encoding("gpt2")

print(generate(model,enc,"First Citizen",100,pa.device))


