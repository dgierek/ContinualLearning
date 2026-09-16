import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNet18_1D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Load standard ResNet-18 backbone (no ImageNet weights)
        self.model = resnet18(weights=None)

        # Modify the first convolution to operate on 1D signals.
        #
        # Original ResNet-18 expects:
        #   Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        #
        # Your CV signal format here seems to be:
        #   (batch, 3, 256)
        #
        # So you reshape it to:
        #   (batch, 3, 1, 256)
        #
        # Then use a Conv2d with kernel height = 1 → effectively 1D convolution.
        self.model.conv1 = nn.Conv2d(
            in_channels=3,      # number of CV channels (you used 3 here)
            out_channels=64,
            kernel_size=(1, 7), # 1D kernel along the width dimension
            stride=(1, 2),
            padding=(0, 3),
            bias=False
        )

        # Replace classifier head
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (batch, 3, 256)
        # Expand to 4D so ResNet can process it:
        # -> (batch, 3, 1, 256)
        x = x.unsqueeze(2)
        return self.model(x)
