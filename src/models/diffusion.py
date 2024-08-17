import torch
from torch import nn
from src.models.unet import UNet, UNetOutputLayer


class Diffusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.time_embedding = TimeEmbedding(dim=320)
        self.unet = UNet()
        self.final = UNetOutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # latent = (batch_size, 4, height / 8, width / 8)
        # context = (batch_size, seq_len, embedding_dim)
        # time = (1, 320)

        time = self.time_embedding(time)
        # time = (1, 1280)

        output = self.unet(latent, context, time)
        # (batch_size, 320, height/8, width/8)

        output = self.final(output)
        # (batch_size, 4, height/8, width/8)

        return output


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dim, 4 * dim)
        self.linear2 = nn.Linear(4 * dim, 4 * dim)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.silu(x)
        x = self.linear2(x)

        return x