import torch
import torch.nn as nn
import torch.nn.functional as F

class CVSmallCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=4):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3   = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(p=0.3)

        self.fc1 = nn.Linear(128 * 1001, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (batch, channels, length)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.dropout(x)

        x = x.view(x.size(0), -1)  # flatten

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CVNanoNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=4):
        super().__init__()

        # Tiny feature extractor
        self.conv1 = nn.Conv1d(in_channels, 8, kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm1d(8)

        self.conv2 = nn.Conv1d(8, 16, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(16)

        self.conv3 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn3   = nn.BatchNorm1d(32)

        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.dropout = nn.Dropout(p=0.3)

        # Compute flatten size automatically
        flatten_size = self._get_flatten_size(in_channels)

        self.fc1 = nn.Linear(flatten_size, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def _get_flatten_size(self, in_channels):
        """Run a dummy forward pass to compute the flattened size."""
        with torch.no_grad():
            x = torch.zeros(1, in_channels, 1001)  # your CV length
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.pool(x)
            return x.numel()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        x = self.dropout(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CVMidNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=4):
        super().__init__()

        # Moderate feature extractor
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm1d(16)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(32)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn3   = nn.BatchNorm1d(64)

        # Pooling to reduce dimensionality
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)

        # Regularization
        self.dropout = nn.Dropout(p=0.3)

        # Compute flatten size automatically
        flatten_size = self._get_flatten_size(in_channels)

        self.fc1 = nn.Linear(flatten_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _get_flatten_size(self, in_channels):
        """Run a dummy forward pass to compute the flattened size."""
        with torch.no_grad():
            x = torch.zeros(1, in_channels, 1001)  # CV length
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.pool(x)
            return x.numel()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        x = self.dropout(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x