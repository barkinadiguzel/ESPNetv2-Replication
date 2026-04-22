import torch.nn as nn
from .depthwise_dilated_conv import DepthwiseDilatedConv
from .group_pointwise_conv import GroupPointwiseConv
from .hff import HFF


class EESPBlock(nn.Module):
    def __init__(self, cin, cout, k=4, g=4):
        super().__init__()

        self.k = k
        mid = cout // k

        self.reduce = GroupPointwiseConv(cin, mid * k, g)

        self.branches = nn.ModuleList([
            DepthwiseDilatedConv(mid, dilation=i + 1)
            for i in range(k)
        ])

        self.fuse = GroupPointwiseConv(mid * k * (k + 1) // 2, cout, g)

        self.hff = HFF()
        self.act = nn.PReLU(cout)

    def forward(self, x):
        x = self.reduce(x)
        xs = torch.chunk(x, self.k, dim=1)

        ys = [b(xs[i]) for i, b in enumerate(self.branches)]
        x = self.hff.fuse(ys)
        x = self.fuse(x)

        return self.act(x)
