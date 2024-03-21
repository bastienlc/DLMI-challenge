"""Simple model based on convolutional neural network
image, annotations = x
image size : 3 x 224 x 224
annotations size : 3
output size : 2

The image is processed by a convolutional neural network, and its output is concatenated with the annotations and processed by a fully connected neural network.
"""

from math import floor

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


class BigCNN(nn.Module):
    def __init__(
        self,
        channels=[6, 12, 24, 48],
        kernels=[5, 5, 5, 5],
        pools=[2, 2, 2, 2],
        strides=[2, 2, 2, 2],
        hidden_dims=[512, 128],
        dropout=0.01,
    ):
        super(BigCNN, self).__init__()

        self.channels = channels
        self.kernels = kernels
        self.pools = pools
        self.strides = strides
        self.hidden_dims = hidden_dims
        self.dropout = nn.Dropout(dropout)

        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.batch_norm = nn.BatchNorm2d(3)
        self.out_size = 254

        for i, (channel, kernel, pool, stride) in enumerate(
            zip(self.channels, self.kernels, self.pools, self.strides)
        ):
            if i == 0:
                self.conv_layers.append(nn.Conv2d(3, channel, kernel))
            else:
                self.conv_layers.append(nn.Conv2d(channels[i - 1], channel, kernel))
            self.pool_layers.append(nn.MaxPool2d(pool, stride=stride))

            # Probably right ?
            self.out_size = (self.out_size - kernel - 1) + 1
            self.out_size = floor((self.out_size - pool - 1) / stride) + 1

        for i, dim in enumerate(self.hidden_dims):
            if i == 0:
                self.linear_layers.append(
                    nn.Linear(
                        self.out_size * self.out_size * channels[-1]
                        + len(CONFIG.cols_annotation),
                        dim,
                    )
                )
            else:
                self.linear_layers.append(nn.Linear(self.hidden_dims[i - 1], dim))

        self.linear_layers.append(nn.Linear(self.hidden_dims[-1], 2))

    def forward(self, x):
        image, annotations = x
        image = self.batch_norm(image)
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            image = pool(F.relu(conv(image)))

        image = image.view(
            image.shape[0], self.out_size * self.out_size * self.channels[-1]
        )
        x = torch.cat((self.dropout(image), annotations), dim=1)

        for layer in self.linear_layers[:-1]:
            x = F.relu(layer(x))

        return self.linear_layers[-1](x)
