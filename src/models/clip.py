import torch
from torch import nn
from attention import SelfAttention


class CLIP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = CLIPEmbedding(voc_size=49408, embedding_dim=768, max_seq_len=77)

        self.layers = [CLIPLayer(n_attention_heads=12, embedding_dim=768) for _ in range(12)]

        self.layernorm = nn.LayerNorm(normalized_shape=768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        # (batch_size, seq_len)
        embedding = self.embedding(tokens)
        # (batch_size, seq_len, embedding_dim)

        for layer in self.layers:
            embedding = layer(embedding)

        # (batch_size, seq_len, embedding_dim)
        output = self.layernorm(embedding)

        return output


class CLIPEmbedding(nn.Module):
    def __init__(self, voc_size: int, embedding_dim: int, max_seq_len: int) -> None:
        super().__init__()

        self.token_embedding = nn.Embedding(num_embeddings=voc_size, embedding_dim=embedding_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(max_seq_len, embedding_dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.token_embedding(tokens)
        token_embeddings += self.positional_encoding

        return token_embeddings


class CLIPLayer(nn.Module):
    def __init__(self, n_attention_heads: int, embedding_dim: int) -> None:
        super().__init__()

        self.layernorm1 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.multihead_attention = SelfAttention(n_heads=n_attention_heads, d_embed=embedding_dim)
        self.layernorm2 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, embedding_dim * 4)
        self.linear2 = nn.Linear(embedding_dim * 4, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # x = (batch_size, seq_len, embedding_dim)
        x = self.layernorm1(x)
        x = self.multihead_attention(x, causal_mask=True)
        x += residual

        residual = x
        x = self.layernorm2(x)
        x = self.linear1(x)
        x = x * torch.sigmoid(1.702 * x)  # QuickGeLU
        x = self.linear2(x)
        x += residual

        return x
