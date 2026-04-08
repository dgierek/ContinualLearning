import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNet18CV(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()

        # Load standard ResNet-18
        self.model = resnet18(weights=None)

        # Replace first conv layer to accept CV channels
        self.model.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Replace classifier head
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)
