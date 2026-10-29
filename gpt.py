from dataclasses import dataclass
from torch import nn


@dataclass
class Config :
    vocab_size = 50_257 # Vocab.
    block_size = 1024 # Context size.
    n_head = 12 # Number of attention heads in multi-head attention.
    n_layer = 12 # Number of transformer layers.
    n_embd = 756 # Embedding size.

class TransformerBlock(nn.Module):
    # TODO: complete
    def __init__(self, config : Config) -> None:
        super().__init__()


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


