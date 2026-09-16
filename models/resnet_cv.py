import torch.nn as nn
from torchvision.models import resnet18

class ResNet18CV(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()

        # Load standard ResNet-18 backbone
        # weights=None → no ImageNet pretraining
        # (CV images differ strongly from natural images)
        self.model = resnet18(weights=None)

        # Replace the first convolution layer so the model accepts
        # 1-channel CV images instead of 3-channel RGB.
        #
        # CV data is typically represented as:
        # - 2D image of current vs potential
        # - or a transformed spectrogram-like representation
        #
        # ResNet expects 3 channels by default, so we override it.
        self.model.conv1 = nn.Conv2d(
            in_channels,   # number of CV channels (usually 1)
            64,            # standard ResNet-18 first layer width
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Replace the classifier head with Dropout + Linear
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.3),  # <-- dropout added here
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Forward pass through the modified ResNet-18
        # x: (batch, 2, 1001)
        # reshape to (batch, 2, 1, 1001)
        x = x.unsqueeze(2)
        return self.model(x)
