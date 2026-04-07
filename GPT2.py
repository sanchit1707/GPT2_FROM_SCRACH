import torch
import torch.nn as nn
import params as pa
import torch.nn.functional as f
import math 

class MaskedAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.w_q=nn.Linear(pa.embed_dim,pa.embed_dim)
        self.w_k=nn.Linear(pa.embed_dim,pa.embed_dim)
        self.w_v=nn.Linear(pa.embed_dim,pa.embed_dim)

        self.w_o=nn.Linear(pa.embed_dim,pa.embed_dim)

    def attention(self,q,k,v):

        def causal_mask(seq_len, device):
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
            return mask
        
        attention_score=q@k.transpose(-1,-2)/math.sqrt(pa.embed_dim//pa.num_heads)

        seq_len=q.size(-2)
        mask=(causal_mask(seq_len,pa.device).unsqueeze(0)).unsqueeze(0)
        

        attention_score.masked_fill_(mask==0,float('-inf'))

        attention_score=attention_score.softmax(dim=-1)

        return attention_score@v
    
    def forward(self,input):
        
        q=self.w_q(input)
        k=self.w_k(input)
        v=self.w_v(input)

        h=pa.num_heads
        d_k=pa.embed_dim//pa.num_heads

        b, s, _ = input.shape

        q=q.view(b,s,h,d_k).transpose(1,2)
        v=v.view(b,s,h,d_k).transpose(1,2)
        k=k.view(b,s,h,d_k).transpose(1,2)

        x=self.attention(q,k,v)

        x=x.transpose(1,2).contiguous().view(b,s,h*d_k)

        return self.w_o(x)
    
class FeedForwardLayer(nn.Module): 
    def __init__(self):
        super().__init__()
        
        self.layer1=nn.Linear(pa.embed_dim,4*pa.embed_dim)
        self.layer2=nn.Linear(4*pa.embed_dim,pa.embed_dim)

    def forward(self,input):

        return self.layer2(f.gelu(self.layer1(input)))
    
class AttentionBlock (nn.Module) :

    def __init__(self):
        super().__init__()

        self.lay1=nn.LayerNorm(pa.embed_dim)
        self.lay2=MaskedAttention()
        self.lay3=nn.LayerNorm(pa.embed_dim)
        self.lay4=FeedForwardLayer()

    def forward(self,input):
        input=input+self.lay2(self.lay1(input))
        input=input+self.lay4(self.lay3(input))

        return input
    
class GPT2(nn.Module):

    def __init__(self):
        super().__init__()

        self.transformer=nn.ModuleDict({
            "tok_embed":nn.Embedding(pa.vocab_size,pa.embed_dim),
            "pos_embed":nn.Embedding(pa.max_seqlen,pa.embed_dim),
            "blocks":nn.ModuleList([AttentionBlock() for _ in range (pa.num_layers)]),
            "lay_nor":nn.LayerNorm(pa.embed_dim)
        })

        self.logits=nn.Linear(pa.embed_dim,pa.vocab_size,bias=False)
        self.logits.weight=self.transformer["tok_embed"].weight

        for module in self.modules():
            if isinstance(module,(nn.Linear,nn.Embedding)):
                module.weight.data.normal_(mean=0.0,std=0.02)
                if isinstance(module,nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()

            elif isinstance(module,nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def forward(self,token_ids):

        b=token_ids.size(0)
        seq_len=token_ids.size(1)

        tokens=self.transformer["tok_embed"](token_ids)
        pos=torch.arange(0,seq_len,device=token_ids.device)
        token_pos=self.transformer["pos_embed"](pos)
        token_pos=token_pos.unsqueeze(0).expand(b,seq_len,-1)
        out=tokens+token_pos

        for block in self.transformer["blocks"]:
            out=block(out)

        out=self.logits(self.transformer["lay_nor"](out))

        return out
    
if __name__=="__main__":
    model=GPT2()
    input=torch.ones((4,9),dtype=torch.long)
    out=model(input)
    print(out.size())






        

             





