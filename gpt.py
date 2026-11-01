from dataclasses import dataclass
from torch import nn
import math
import torch
import torch.nn.functional as F

@dataclass
class Config :
    vocab_size = 50_257 # Vocab.
    block_size = 1024 # Context size.
    n_head = 12 # Number of attention heads in multi-head attention.
    n_layer = 12 # Number of transformer layers.
    n_embd = 768 # Embedding size.


class CausalSelfAttention(nn.Module):
    bias: torch.Tensor

    def __init__(self, config : Config) -> None:
        super().__init__()

        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.register_buffer("bias",
            torch.tril(
                torch.ones(config.block_size, config.block_size).view(
                1, config.block_size, config.block_size
                )
            )
        )


    def forward(self, x):
        B, T, E = x.size()

        Q = self.q_proj(x) # [B,T,E]
        K = self.k_proj(x) # [B,T,E]
        V = self.v_proj(x) # [B,T,E]

        A = (Q @ K.transpose(-2, -1)) * (1 / math.sqrt(K.size(-1)))
        A = A.masked_fill(self.bias[:, :T, :T] == 0, float('-inf'))
        A = F.softmax(A, dim=-1)

        y = A @ V
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config : Config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class TransformerBlock(nn.Module):  

    def __init__(self, config : Config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.c_attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.c_attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self, config : Config) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
                h  = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)],
            ),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

print (GPT(Config()))