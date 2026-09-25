"""Small square-token attention blocks for the B2 controlled architecture arm."""
import torch
from torch import nn


class SquareAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        if channels % 4:
            raise ValueError('Square attention requires channels divisible by four')
        self.heads = 4
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.projection = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(nn.Linear(channels, 2 * channels), nn.GELU(),
                                 nn.Linear(2 * channels, channels))

    def forward(self, x):
        n, channels, _, _ = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        qkv = self.qkv(self.norm1(tokens)).reshape(n, 64, 3, self.heads, channels // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # Explicit matmul avoids train/eval fused-kernel dispatch differences.
        scores = (q @ k.transpose(-2, -1)) * ((channels // self.heads) ** -0.5)
        mixed = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(n, 64, channels)
        tokens = tokens + self.projection(mixed)
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(n, channels, 8, 8)
