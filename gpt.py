from dataclasses import dataclass
from torch import nn
import math
import torch
import torch.nn.functional as F

@dataclass
class Config :
    vocab_size : int = 50_257 # Vocab.
    block_size : int = 1024 # Context size.
    n_head : int = 12 # Number of attention heads in multi-head attention.
    n_layer : int = 12 # Number of transformer layers.
    n_embd : int = 768 # Embedding size.


class CausalSelfAttention(nn.Module):
    bias: torch.Tensor

    def __init__(self, config : Config) -> None:
        super().__init__()

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.register_buffer("bias",
            torch.tril(
                torch.ones(config.block_size, config.block_size).view(
                1, config.block_size, config.block_size
                )
            )
        )

    def forward(self, x):
        B, T, E = x.size() # Batch, Tokens, Embeddings

        qkv = self.c_attn(x) # [B, T, 3E]
        Q, K, V = qkv.split(self.n_embd, dim=-1) # [B, T, E] each

        A = (Q @ K.transpose(-2, -1)) * (1 / math.sqrt(K.size(-1)))
        A = A.masked_fill(self.bias[:, :T, :T] == 0, float('-inf'))
        A = F.softmax(A, dim=-1) # [B, nh, T, T]

        y = A @ V # [B, nh, T, e]
        y = y.transpose(1,2).contiguous().view(B,T,E) # [B, T, E]
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
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
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
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = Config(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

# ---
def test_loading_works():
    GPT.from_pretrained("gpt2")
    print('Successs !')

def test_attention_runs():
    mock_tensor = torch.randn([4, Config().block_size, Config().n_embd])
    print(CausalSelfAttention(Config())(mock_tensor).shape)
    print('Success !')
# ---
