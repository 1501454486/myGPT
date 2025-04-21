from turtle import pos
import torch
import torch.nn as nn
from .configs.gpt_config_124m import GPT_CONFIG_124M
from .blocks.TransformerBlock import TransformerBlock
from .blocks.LayerNormBlock import LayerNorm


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.transformerBlocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias = False
        )

    def forward(self, in_idx):
        batch_size, num_tokens = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(in_idx)

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.transformerBlocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        
        return logits