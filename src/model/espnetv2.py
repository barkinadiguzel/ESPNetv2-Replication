import torch.nn as nn
from ..modules.stem import Stem
from ..modules.esp_stage import ESPStage


class ESPNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.stem = Stem(3, 16)

        self.stage1 = ESPStage(16, 32, 1, True)
        self.stage2 = ESPStage(32, 64, 3, True)
        self.stage3 = ESPStage(64, 128, 7, True)
        self.stage4 = ESPStage(128, 256, 3, True)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.head(x)

        return x.flatten(1)
