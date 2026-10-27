"""PANNs Cnn14_16k — только инференс.

Перенесено из github.com/qiuqiangkong/audioset_tagging_cnn (pytorch/models.py, коммит d2f4b8c),
Copyright (c) 2018-2020 Qiuqiang Kong, лицензия MIT (models/panns/LICENSE-code.MIT).

Изменения относительно оригинала: убраны SpecAugmentation, mixup и dropout (нужны только
для обучения). Имена слоёв не менялись, чтобы веса загружались строго (strict=True).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import LogmelFilterBank, Spectrogram

# Параметры Cnn14_16k из README авторов
SAMPLE_RATE = 16000
WINDOW_SIZE = 512
HOP_SIZE = 160
MEL_BINS = 64
FMIN = 50
FMAX = 8000
CLASSES_NUM = 527


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, pool_size=(2, 2)) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, kernel_size=pool_size)


class Cnn14_16k(nn.Module):  # имя как у авторов
    def __init__(self):
        super().__init__()
        self.spectrogram_extractor = Spectrogram(
            n_fft=WINDOW_SIZE, hop_length=HOP_SIZE, win_length=WINDOW_SIZE, window="hann",
            center=True, pad_mode="reflect", freeze_parameters=True,
        )
        self.logmel_extractor = LogmelFilterBank(
            sr=SAMPLE_RATE, n_fft=WINDOW_SIZE, n_mels=MEL_BINS, fmin=FMIN, fmax=FMAX,
            ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True,
        )
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, CLASSES_NUM, bias=True)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (batch, samples) float32 16 кГц -> вероятности классов (batch, 527)."""
        x = self.spectrogram_extractor(audio)  # (batch, 1, time, freq)
        x = self.logmel_extractor(x)  # (batch, 1, time, mel)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x, pool_size=(1, 1))
        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = F.relu_(self.fc1(x1 + x2))
        return torch.sigmoid(self.fc_audioset(x))
