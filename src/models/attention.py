import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias: bool = True, out_proj_bias: bool = True):
        super().__init__()

        self.in_proj = nn.Linear(in_features=d_embed, out_features=3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(in_features=d_embed, out_features=d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask: bool = False) -> torch.Tensor:
        # x = (batch_size, seq_len, dim)
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        intermin_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # (batch_size, seq_len, dim)
        x = self.in_proj(x)

        # (batch_size, seq_len, dim * 3)
        q, k, v = x.chunk(3, dim=-1)
        # q = (batch_size, seq_len, dim)
        # k = (batch_size, seq_len, dim)
        # v = (batch_size, seq_len, dim)

        q = q.view(intermin_shape).transpose(1, 2)
        k = k.view(intermin_shape).transpose(1, 2)
        v = v.view(intermin_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        output = weight @ v

        output = output.transpose(1, 2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        return output
