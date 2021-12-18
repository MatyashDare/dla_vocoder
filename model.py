import math
import torch
from torch import nn
import torch.nn.functional as F

PADDING_MODE = 'same'
RELUSLOPE = 0.1


class Block(nn.Module):
    def __init__(self, c_in, kernel_size):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(c_in, c_in, kernel_size=kernel_size, dilation=1, padding='same'),
            nn.LeakyReLU(RELUSLOPE),
            nn.Conv1d(c_in, c_in, kernel_size=kernel_size, dilation=3, padding='same'),
            nn.LeakyReLU(RELUSLOPE),
            nn.Conv1d(c_in, c_in, kernel_size=kernel_size, dilation=5, padding='same'),
            nn.LeakyReLU(RELUSLOPE),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(c_in, c_in, kernel_size=kernel_size, dilation=1, padding='same'),
            nn.LeakyReLU(RELUSLOPE),
            nn.Conv1d(c_in, c_in, kernel_size=kernel_size, dilation=1, padding='same'),
            nn.LeakyReLU(RELUSLOPE),
            nn.Conv1d(c_in, c_in, kernel_size=kernel_size, dilation=1, padding='same'),
            nn.LeakyReLU(RELUSLOPE),
        )

    def forward(self, x):
        return self.conv1(x) + self.conv2(x)


class Resblocks(nn.Module):
    def __init__(self, c_in, resblocks_kernels):
        super().__init__()
        self.blocks = nn.ModuleList([Block(c_in, kernel_size) for kernel_size in resblocks_kernels])

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return x

class HIFIGenerator(nn.Module):
    def __init__(self,):
        super().__init__()
        first_channels = 128
        self.preconv = nn.Sequential(nn.Conv1d(80, first_channels, kernel_size=7, padding=PADDING_MODE), nn.LeakyReLU(RELUSLOPE))
        self.transposes = nn.ModuleList([])
        self.resblocks = nn.ModuleList([])
        kernels = [16, 16, 4, 4]
        resblocks_kernels = [3, 7, 11]
        c = first_channels
        for i in range(len(kernels)):
            self.transposes.append(nn.ConvTranspose1d(c, c // 2, kernel_size=kernels[i], stride=kernels[i] // 2, padding=kernels[i] // 4))
            self.resblocks.append(nn.Sequential(Resblocks(c // 2, resblocks_kernels), nn.LeakyReLU(RELUSLOPE)))
            c = c // 2
        self.postconv = nn.Sequential(nn.Conv1d(c, 1, kernel_size=7, padding=PADDING_MODE), nn.Tanh())

    def forward(self, x):
        x = self.preconv(x)
        for i in range(len(self.transposes)):
            x = self.transposes[i](x)
            x = self.resblocks[i](x)
        return self.postconv(x)

