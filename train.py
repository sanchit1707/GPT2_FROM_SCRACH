import torch
import torch.nn as nn
import torch.nn.functional as f
import tiktoken
import params as pa
import torch.optim as Adamo
from GPT2 import GPT2

with open('./shakespeare.txt') as file:
    input=file.read()

enc=tiktoken.get_encoding('gpt2')
tokens=torch.tensor(enc.encode(input)).type('torch.LongTensor')
decoded=enc.decode(list(tokens))
assert input == decoded

def get_batch(tokens, batch_size, block_size, device):
    ix = torch.randint(0, len(tokens) - block_size - 1, (batch_size,))
    
    x = torch.stack([tokens[i:i+block_size] for i in ix])
    y = torch.stack([tokens[i+1:i+block_size+1] for i in ix])
    
    return x.to(device), y.to(device)


model=GPT2()
model.to(pa.device)
optim=Adamo.AdamW(model.parameters(),lr=3e-4)
loss_fn=torch.nn.CrossEntropyLoss()

for i in range(2000):
    x, y = get_batch(tokens, pa.batch_size, pa.max_seqlen, pa.device)
    
    logits = model(x)
    loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
    
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    if i % 100 == 0:
        print(f"step {i}: loss = {loss.item():.4f}")
