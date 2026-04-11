import torch

vocab_size=50257
max_seqlen=256
num_heads=12
batch_size=2
embed_dim=768
num_layers=12


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


