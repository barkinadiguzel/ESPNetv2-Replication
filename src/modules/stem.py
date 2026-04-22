import torch.nn as nn


class Stem:
    def __init__(self, cin=3, cout=16):
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.PReLU(cout)
        )

    def __call__(self, x):
        return self.block(x)
