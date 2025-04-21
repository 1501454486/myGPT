from matplotlib.patheffects import Normal
from sympy import Mul
import torch
import torch.nn as nn
from .FeedForwardBlock import FeedForward
from .MultiHeadAttentionBlock import MultiHeadAttention
from .LayerNormBlock import LayerNorm


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            dropout = cfg["drop_rate"],
            num_heads = cfg["n_heads"],
            qkv_bias = cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        x = self.drop_shortcut(self.attn(self.norm1(x))) + x
        x = self.drop_shortcut(self.ff(self.norm2(x))) + x
        return x