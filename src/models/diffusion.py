import torch
from torch import nn
import torch.nn.functional as F
from attention import SelfAttention, CrossAttention


class Diffusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.time_embedding = TimeEmbedding(dim=320)
        self.unet = UNet()
        self.final = UNetOutputLayer(128, 4)

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


class UNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = [
            # (batch_size, 4, height/8, width/8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(UNetResidualBlock(320, 320), UNetAttentionBlock(8, 40)),
            SwitchSequential(UNetResidualBlock(320, 320), UNetAttentionBlock(8, 40)),
            # (batch_size, 320, height/8, width/8)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNetResidualBlock(320, 640), UNetAttentionBlock(8, 80)),
            SwitchSequential(UNetResidualBlock(640, 640), UNetAttentionBlock(8, 80)),
            # (batch_size, 640, height/16, width/16)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNetResidualBlock(640, 1280), UNetAttentionBlock(8, 160)),
            SwitchSequential(UNetResidualBlock(1280, 1280), UNetAttentionBlock(8, 160)),
            # (batch_size, 1280, height/32, width/32)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNetResidualBlock(1280, 1280), UNetAttentionBlock(8, 160)),
            SwitchSequential(UNetResidualBlock(1280, 1280), UNetAttentionBlock(8, 160))
            # (batch_size, 1280, height/64, width/64)
        ]

        self.bottleneck = [
            UNetResidualBlock(1280, 1280),
            UNetAttentionBlock(8, 160),
            UNetResidualBlock(1280, 1280)
        ]

        self.decoder = [
            # (batch_size, 2560, height/64, width/64)
            SwitchSequential(UNetResidualBlock(2560, 1280)),
            SwitchSequential(UNetResidualBlock(1280, 1280)),
            SwitchSequential(UNetResidualBlock(1280, 1280), UpSample(1280)),
            SwitchSequential(UNetResidualBlock(2560, 1280), UNetAttentionBlock(8, 160)),
            SwitchSequential(UNetResidualBlock(2560, 1280), UNetAttentionBlock(8, 160)),
            SwitchSequential(UNetResidualBlock(1920, 1280), UNetAttentionBlock(8, 160), UpSample(1280)),
        ]


class SwitchSequential(nn.Module):
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNetAttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNetResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)

        return x


class UNetResidualBlock(nn.Module):
    def __init__(self)


class UNetAttentionBlock(nn.Module):
    def __init__(self)
