"""DS-CNN для поиска ключевых слов (Zhang et al., «Hello Edge: Keyword Spotting on
Microcontrollers», 2017): свёртка + блоки depthwise-separable свёрток + усреднение + линейный слой.
"""

import torch
from torch import nn


class DSBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.dw(x)))
        return torch.relu(self.bn2(self.pw(x)))


class DSCNN(nn.Module):
    def __init__(self, n_classes: int, n_mels: int = 40, channels: int = 64, blocks: int = 4):
        super().__init__()
        self.bn_in = nn.BatchNorm2d(1)  # нормировка признаков без внешней статистики
        self.conv = nn.Conv2d(1, channels, (10, 4), stride=(2, 2), padding=(5, 1), bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.blocks = nn.Sequential(*[DSBlock(channels) for _ in range(blocks)])
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, кадры, mel) -> логиты (batch, классы)."""
        x = self.bn_in(x.unsqueeze(1))
        x = torch.relu(self.bn(self.conv(x)))
        x = self.blocks(x)
        x = x.mean(dim=(2, 3))
        return self.fc(self.dropout(x))
