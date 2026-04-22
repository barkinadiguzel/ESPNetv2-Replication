import torch.nn as nn
from ..blocks.eesp_block import EESPBlock
from ..blocks.strided_eesp_block import StridedEESPBlock


class ESPStage(nn.Module):
    def __init__(self, cin, cout, repeats, first_down=False, k=4, g=4):
        super().__init__()

        layers = []

        if first_down:
            layers.append(StridedEESPBlock(cin, cout, k, g))
            cin = cout

        for _ in range(repeats):
            layers.append(EESPBlock(cin, cout, k, g))
            cin = cout

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
