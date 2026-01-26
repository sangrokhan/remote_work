
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network for demonstration purposes.
    It is designed to be instantiated locally without downloading weights.
    Structure intentionally follows some ResNet naming conventions (conv1, fc)
    to remain compatible with the existing transfer_learning utilities.
    """
    def __init__(self, num_classes=1000):
        super(SimpleCNN, self).__init__()
        # First layer named 'conv1' to match transfer_utils expectations
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64)
        self.layer2 = self._make_layer(64, 128)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Final layer named 'fc' to match transfer_utils expectations
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
