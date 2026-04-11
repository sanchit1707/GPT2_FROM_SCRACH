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

def generate(model, enc, prompt, max_new_tokens=50, device="cuda"):
    model.eval()
    
    # encode input
    tokens = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    
    for _ in range(max_new_tokens):
        # crop if too long
        tokens_cond = tokens[:, -pa.max_seqlen:]
        
        with torch.no_grad():
            logits = model(tokens_cond)
        
        # take last token prediction
        logits = logits[:, -1, :]
        
        # convert to probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # sample next token 
        next_token = torch.multinomial(probs, num_samples=1)
        
        # append
        tokens = torch.cat((tokens, next_token), dim=1)
    
    # decode back to text
    return enc.decode(tokens[0].tolist())
if __name__ == "__main__":
    for i in range(50000):
        x, y = get_batch(tokens, pa.batch_size, pa.max_seqlen, pa.device)
        
        logits = model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        if i % 10 == 0:
            print(f"step {i}: loss = {loss.item():.4f}")
        if i % 500 == 0:
            torch.save(model.state_dict(), f"checkpoints/checkpoint_{i}.pth")
        
            
        
            text = generate(model, enc, prompt="To be or not to be", device=pa.device)
            print("Generated:", text)
            print("-" * 50)

torch.save(model.state_dict(), "gpt2_model.pth")
if __name__ == "__main__":
    text = generate(model, enc, prompt="To be or not to be", device=pa.device)
    print("Generated:", text)   
