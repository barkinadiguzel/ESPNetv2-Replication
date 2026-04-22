import torch
import torch.nn as nn

class DepthwiseDilatedConv(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, stride=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=False
        )

        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
