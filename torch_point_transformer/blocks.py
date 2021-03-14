import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        bias: bool = False,
        bn: bool = True,
        activation: bool = True,
    ):
        super().__init__()
        layers = [
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=bias,
            )
        ]

        if bn:
            layers.append(nn.BatchNorm1d(out_channels))

        if activation:
            layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FCBlock(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bn: bool = True,
        activation: bool = True,
        p: float = 0.0,
    ):
        super().__init__()
        layers = [
            nn.Linear(
                in_features=in_features,
                out_features=out_features,
            )
        ]

        if p != 0.0:
            layers.append(nn.Dropout(p))

        if bn:
            layers.append(nn.BatchNorm1d(out_features))

        if activation:
            layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PointTransformerLayer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class PointTransformerBlock(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.model = nn.Sequential(
            ConvBlock(in_channels=dim, out_channels=dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class TransitionDownBlock(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class TransitionUpBlock(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


