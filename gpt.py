from dataclasses import dataclass
from torch import nn
import inspect
import math
import time
import torch
import torch.nn.functional as F

@dataclass
class Config:
    vocab_size: int = 50_257  # Vocab.
    block_size: int = 1024  # Context size.
    n_head: int = 12  # Number of attention heads in multi-head attention.
    n_layer: int = 12  # Number of transformer layers.
    n_embd: int = 768  # Embedding size.


class CausalSelfAttention(nn.Module):
    bias: torch.Tensor

    def __init__(self, config: Config) -> None:
        super().__init__()

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.GPT2_SCALE_INIT = 1

        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones(config.block_size, config.block_size).view(
                    1, config.block_size, config.block_size
                )
            ),
        )

    def forward(self, x):
        B, T, E = x.size()  # Batch, Tokens, Embeddings

        qkv = self.c_attn(x)  # [B, T, 3E]
        Q, K, V = qkv.split(self.n_embd, dim=-1)  # [B, T, E] each

        Q = Q.view(B, T, self.n_head, E // self.n_head)  # [B,T,nh,E']
        Q = Q.transpose(1, 2)  # [B,nh, T, E']
        K = K.view(B, T, self.n_head, E // self.n_head)  # [B,T,nh,E']
        K = K.transpose(1, 2)  # [B,nh, T, E']
        V = V.view(B, T, self.n_head, E // self.n_head)  # [B,T,nh,E']
        V = V.transpose(1, 2)  # [B,nh, T, E']

        # A = (Q @ K.transpose(-2, -1)) * (1 / math.sqrt(K.size(-1)))
        # A = A.masked_fill(self.bias[:, :T, :T] == 0, float("-inf"))
        # A = F.softmax(A, dim=-1)  # [B, nh, T, T]
        # y = A @ V  # [B, nh, T, E']

        y = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, E)  # [B, T, E]
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.GPT2_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class TransformerBlock(nn.Module):

    def __init__(self, config: Config) -> None:
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

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList(
                    [TransformerBlock(config) for _ in range(config.n_layer)],
                ),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Init weights std=0.02 and bias=0.
        # Divide std by sqrt(n_layers) to account for residual stream variance.
        # No need to update layer norm. Defaults are offset=0 and scale=1.
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            std = 0.02
            if hasattr(module, "GPT2_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        B, T = x.size()
        assert T <= self.config.block_size

        pos = torch.arange(0, T, dtype=torch.long, device=x.device)

        x = self.transformer.wte(x) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = Config(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
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

    def configure_optimizers(self, weight_decay, learning_rate, device):
        """Sets up optimizer which applies weight decay to non 1d tensors."""

        # All of the candidate parameters (that require grad).
        param_dict = {pn: p for pn, p in self.named_parameters ()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # Any parameters that is 2D will be weight decayed.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items () if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items () if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay}, 
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # Create AdamW optimizer and use the fused version if it is available.
        use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters and 'cuda' in device
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

        print (f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} params")
        print (f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} params")
        print (f"using fused AdamW: {use_fused}")
        
        return optimizer

import tiktoken


class DataLoader:
    def __init__(self, B, T) -> None:
        self.B, self.T = B, T
        with open("tinyshakespeare.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        print(f"Loaded {len(tokens)} tokens")
        print(f"1 epoch = {len(tokens)// (B*T)} batches")
        self.current_pos = 0

    def next_batch(self):
        B, T = self.B, self.T
        buffer = self.tokens[self.current_pos : self.current_pos + (B * T) + 1]
        x = buffer[:-1].view(B, T)
        y = buffer[1:].view(B, T)
        self.current_pos += B * T

        # Reset current_pos when it overflows.
        if self.current_pos + (B * T + 1) >= len(self.tokens):
            self.current_pos = 0

        return x, y


# ---
# Environment Setup
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(s: int):
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.mps.manual_seed(s)


device = get_device()
print(f"Using device: {device}")


# ################
# # INFERENCE
# ################
def run_inference():
    set_seed(42)
    ### Tokenizer Setup
    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = tokenizer.encode("Hello, I'm a language model,")

    ### Generate Text Loop
    max_output_length = 30
    num_return_sequences = 5
    tokens = torch.tensor(tokens, dtype=torch.long)  # [4]
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # [5,4]
    x = tokens.to(device)

    # Load model
    # model = GPT.from_pretrained("gpt2")
    model = GPT(Config(vocab_size=50_304)).to(device)
    model.eval()

    # Run inference.
    with torch.no_grad():
        while x.size(1) < max_output_length:
            logits, loss = model(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            x_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, x_next], dim=1)

    for row in x.tolist():
        print(tokenizer.decode(row))


# run_inference()

################
# TRAINING
################


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    elif device == "cpu":
        torch.cpu.synchronize()


# Load dataloader.
dataloader = DataLoader(B=32, T=1024)
torch.set_float32_matmul_precision("high")

# Load model.
model = GPT(Config(vocab_size=50_304))
model.to(device)
model = torch.compile(model)

# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)#, betas=[0.9, 0.95], eps=1e-8)
learning_rate = 6e-4
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=learning_rate, device=device)

num_steps = 10
sum_tokens_per_sec = 0
sum_dt = 0


def get_lr(it, max_steps, max_lr):
    """ Cosine with warmup."""

    min_lr = max_lr * 0.1
    warmup_steps = max_steps * 0.1

    # Warmup.
    if it < warmup_steps: return max_lr * (it+1) / warmup_steps
    # Min LR.
    if it > num_steps: return min_lr
    # Cosine.
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 + 0.5 * (math.cos(math.pi * decay_ratio)) # [0,1]
    return coeff * (max_lr-min_lr) + min_lr

for step in range(num_steps):
    t0 = time.time()
    optimizer.zero_grad()
    x, y = dataloader.next_batch()
    x, y = x.to(device), y.to(device)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step, num_steps, learning_rate)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    synchronize(device)
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = dataloader.B * dataloader.T / (t1 - t0)
    print(f"Step {step}, loss {loss.item()}, lr: {lr:.4e} norm: {norm:.4f} dt: {dt:.2f} ms, tok.sec: {tokens_per_sec:.2f}")

    if step == 0 : continue
    sum_dt += dt
    sum_tokens_per_sec += tokens_per_sec
print(f"avg_dt: {sum_dt/(num_steps-1):.2f} ms, tok.sec: {sum_tokens_per_sec/(num_steps-1):.2f}")