import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(torch.nn.Module):
    """
    A CNN model for image classification
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """
        Initialize the CNN
        Arguments:
            num_channels: The number of input image channels.
            num_classes: The number of output classes.
        """
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 6, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=2)
        self.bn = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        Arguments:
            x: The input data.
        Returns:
            The output of the network.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.bn(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
