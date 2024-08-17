import torch
from torch import nn
import torch.nn.functional as F
from src.models.attention import SelfAttention, CrossAttention


class UpSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, features, height, width)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        # (batch_size, features, height * 2, width * 2)
        return self.conv(x)


class UNetOutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.silu = nn.SiLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, 320, height/8, width/8)
        x = self.groupnorm(x)
        # (batch_size, 320, height/8, width/8)
        x = self.silu(x)
        # (batch_size, 320, height/8, width/8)
        x = self.conv(x)
        # (batch_size, 4, height/8, width/8)
        return x


class SwitchSequential(nn.Sequential):
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
    def __init__(self, in_channels: int, out_channels: int, n_time=1280):
        super().__init__()
        self.groupnorm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear = nn.Linear(n_time, out_channels)

        self.groupnorm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.silu = nn.SiLU()

        if in_channels != out_channels:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.residual_layer = nn.Identity()

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # x = (batch_size, in_channels, height, width)
        # time = (1, 1280)

        residual = x

        x = self.groupnorm1(x)
        x = self.silu(x)
        x = self.conv1(x)

        time = self.silu(time)
        time = self.linear(time)
        merged = x + time.unsqueeze(-1).unsqueeze(-1)

        merged = self.groupnorm2(merged)
        merged = self.silu(merged)
        merged = self.conv2(merged)

        residual = self.residual_layer(residual)

        return merged + residual


class UNetAttentionBlock(nn.Module):
    def __init__(self, n_heads: int, n_embed: int, context_dim: int = 768):
        super().__init__()
        channels = n_heads * n_embed

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm1 = nn.LayerNorm(channels)
        self.attention1 = SelfAttention(n_heads, channels, in_proj_bias=False)
        self.layernorm2 = nn.LayerNorm(channels)
        self.attention2 = CrossAttention(n_heads, channels, context_dim, in_proj_bias=False)
        self.layernorm3 = nn.LayerNorm(channels)
        self.linear1 = nn.Linear(channels, 4 * channels * 2)
        self.linear2 = nn.Linear(channels * 4, channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x = (batch_size, features, height, width)
        # context = (batch_size, seq_len, dim)
        residual_long = x

        # (batch_size, features, height, width)
        x = self.groupnorm(x)
        # (batch_size, features, height, width)
        x = self.conv1(x)
        # (batch_size, features, height, width)

        n, c, h, w = x.shape

        # (batch_size, features, height, width)
        x = x.view((n, c, h * w))
        # (batch_size, features, height * width)
        x = x.transpose(-1, -2)
        # (batch_size, height * width, transpose)

        residual_short = x

        x = self.layernorm1(x)
        x = self.attention1(x)
        x += residual_short

        residual_short = x

        x = self.layernorm2(x)
        x = self.attention2(x, context)
        x += residual_short

        residual_short = x

        x = self.layernorm3(x)
        x, gate = self.linear1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)

        x = self.linear2(x)

        x += residual_short
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))
        x = self.conv2(x)

        return x + residual_long


class UNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.ModuleList([
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
            SwitchSequential(UNetResidualBlock(1280, 1280)),
            SwitchSequential(UNetResidualBlock(1280, 1280))
            # (batch_size, 1280, height/64, width/64)
        ])

        self.bottleneck = SwitchSequential(
            UNetResidualBlock(1280, 1280),
            UNetAttentionBlock(8, 160),
            UNetResidualBlock(1280, 1280)
        )

        self.decoder = nn.ModuleList([
            # (batch_size, 2560, height/64, width/64)
            SwitchSequential(UNetResidualBlock(2560, 1280)),
            SwitchSequential(UNetResidualBlock(2560, 1280)),
            SwitchSequential(UNetResidualBlock(2560, 1280), UpSample(1280)),
            SwitchSequential(UNetResidualBlock(2560, 1280), UNetAttentionBlock(8, 160)),
            SwitchSequential(UNetResidualBlock(2560, 1280), UNetAttentionBlock(8, 160)),
            SwitchSequential(UNetResidualBlock(1920, 1280), UNetAttentionBlock(8, 160), UpSample(1280)),
            SwitchSequential(UNetResidualBlock(1920, 640), UNetAttentionBlock(8, 80)),
            SwitchSequential(UNetResidualBlock(1280, 640), UNetAttentionBlock(8, 80)),
            SwitchSequential(UNetResidualBlock(960, 640), UNetAttentionBlock(8, 80), UpSample(640)),
            SwitchSequential(UNetResidualBlock(960, 320), UNetAttentionBlock(8, 40)),
            SwitchSequential(UNetResidualBlock(640, 320), UNetAttentionBlock(8, 40)),
            SwitchSequential(UNetResidualBlock(640, 320), UNetAttentionBlock(8, 40))
        ])

    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        for layer in self.encoder:
            x = layer(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layer in self.decoder:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layer(x, context, time)

        return x
