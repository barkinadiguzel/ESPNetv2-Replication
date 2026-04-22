import torch.nn as nn

class GroupPointwiseConv(nn.Module):
    def __init__(self, in_ch, out_ch, groups=4):
        super().__init__()

        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=1,
            groups=groups,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
