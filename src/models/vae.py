import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.attention import SelfAttention


class VAEEncoder(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            # (batch_size, 3, height, width)
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1),

            # (batch_size, 128, height, width), height and width unchanged due to padding = 1 and stride = 1
            VAEResidualBlock(in_channels=128, out_channels=128),

            # (batch_size, 128, height, width)
            VAEResidualBlock(in_channels=128, out_channels=128),

            # (batch_size, 128, height, width)
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=0),

            # (batch_size, 128, height/2, width/2), height and width halved due to stride = 2
            VAEResidualBlock(in_channels=128, out_channels=256),

            # (batch_size, 256, height/2, width/2)
            VAEResidualBlock(in_channels=256, out_channels=256),

            # (batch_size, 256, height/2, width/2)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=0),

            # (batch_size, 256, height/4, width/4) due to stride = 2 and padding = 0
            VAEResidualBlock(in_channels=256, out_channels=512),

            # (batch_size, 512, height/4, width/4)
            VAEResidualBlock(in_channels=512, out_channels=512),

            # (batch_size, 512, height/4, width/4)
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0),

            # (batch_size, 512, height/8, width/8) due to stride = 2 and padding = 0
            VAEResidualBlock(in_channels=512, out_channels=512),

            # (batch_size, 512, height/8, width/8)
            VAEResidualBlock(in_channels=512, out_channels=512),

            # (batch_size, 512, height/8, width/8)
            VAEResidualBlock(in_channels=512, out_channels=512),

            # (batch_size, 512, height/8, width/8)
            VAEAttentionBlock(channels=512),

            # (batch_size, 512, height/8, width/8)
            VAEResidualBlock(in_channels=512, out_channels=512),

            # (batch_size, 512, height/8, width/8)
            nn.GroupNorm(num_groups=32, num_channels=512),

            # (batch_size, 512, height/8, width/8)
            nn.SiLU(),

            # (batch_size, 512, height/8, width/8)
            nn.Conv2d(in_channels=512, out_channels=8, kernel_size=3, padding=1),  # bottleneck

            # (batch_size, 8, height/8, width/8)
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, padding=0),
        )
        # (batch_size, 8, height/8, width/8)

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x = (batch_size, 3, height, width)
        # noise = (batch_size, out_channels, height/8, width/8)

        for layer in self:
            if getattr(layer, "stride", None) == (2, 2):
                # for layers with stride 2, apply asymmetrical padding only on right and bottom
                x = F.pad(x, (0, 1, 0, 1))
            x = layer(x)

        # compute mean and log variance as chunks of the output
        # x = (batch_size, 8, height/8, width/8)
        mean, log_variance = torch.chunk(x, chunks=2, dim=1)
        # mean = (batch_size, 4, height/8, width/8)
        # log_variance = (batch_size, 4, height/8, width/8)

        # clamp the values of the variance into (-30, 20) range
        log_variance = torch.clamp(log_variance, min=-30, max=20)

        # transform log variance into variance
        variance = log_variance.exp()

        # compute standard deviation from variance
        st_dev = variance.sqrt()

        # sample from the noise distribution by converting the input distribution to the noise distribution
        x = mean + st_dev * noise

        # scale the output by a constant
        x *= 0.18215

        return x


class VAEResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)

        # define decoder layers
        self.groupnorm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

        self.groupnorm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)

        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = (batch_size, in_channels, height, width)
        residual = x

        x = self.groupnorm1(x)
        x = self.silu(x)
        x = self.conv1(x)

        x = self.groupnorm2(x)
        x = self.silu(x)
        x = self.conv2(x)

        residual = self.residual(residual)

        x += residual
        return x


class VAEAttentionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.groupnorm1 = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.selfattention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = (batch_size, features, height, width)
        residual = x

        x = self.groupnorm1(x)

        batch_size, features, height, width = x.shape

        # (batch_size, features, height, width)
        x = x.view(batch_size, features, height * width)
        # (batch_size, features, height * width)
        x = x.transpose(-1, -2)
        # (batch_size, height * width, features)
        x = self.selfattention(x)
        # (batch_size, height * width, features)
        x = x.transpose(-1, -2)
        # (batch_size, features, height * width)
        x = x.view(batch_size, features, height, width)
        # (batch_size, features, height, width)

        x += residual

        return x


class VAEDecoder(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            VAEResidualBlock(512, 512),
            VAEAttentionBlock(512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAEResidualBlock(512, 256),
            VAEResidualBlock(256, 256),
            VAEResidualBlock(256, 256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            VAEResidualBlock(256, 128),
            VAEResidualBlock(128, 128),
            VAEResidualBlock(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = (batch_size, 4, height/8, width/8)
        x /= 0.18215

        for layer in self:
            x = layer(x)

        return x
