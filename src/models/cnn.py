"""Simple model based on convolutional neural network
image, annotations = x
image size : 3 x 224 x 224
annotations size : 3
output size : 2

The image is processed by a convolutional neural network, and its output is concatenated with the annotations and processed by a fully connected neural network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CONFIG


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53 + len(CONFIG.cols_annotation), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        image, annotations = x
        image = self.pool(F.relu(self.conv1(image)))
        image = self.pool(F.relu(self.conv2(image)))
        image = image.view(-1, 16 * 53 * 53)
        x = torch.cat((image, annotations), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
