import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNet18_1D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Load standard ResNet-18
        self.model = resnet18(weights=None)
        # CHECK 1D MODELS (torchaudio)
        # Modify first conv to accept 1x256 "images"
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=(1, 7),
            stride=(1, 2),
            padding=(0, 3),
            bias=False
        )

        # Replace classifier
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (batch, 3, 256)
        x = x.unsqueeze(2)  # -> (batch, 3, 1, 256)
        return self.model(x)
