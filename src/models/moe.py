from collections import OrderedDict

import torch
import torch.nn as nn
from torch_scatter import scatter_mean

from ..config import CONFIG


class ConvolutionalNetwork(nn.Module):
    def __init__(self, in_channels=3, init_features=8):
        super(ConvolutionalNetwork, self).__init__()

        features = init_features
        self.encoder1 = ConvolutionalNetwork._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = ConvolutionalNetwork._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = ConvolutionalNetwork._block(
            features * 2, features * 4, name="enc3"
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = ConvolutionalNetwork._block(
            features * 4, features * 8, name="enc4"
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder5 = ConvolutionalNetwork._block(
            features * 8, features * 16, name="enc5"
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))

        return enc5

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


class GatingNetwork(nn.Module):
    def __init__(self, maps, width, height):
        super(GatingNetwork, self).__init__()
        self.conv = nn.Conv2d(maps, 1, kernel_size=1, padding=0, stride=1)
        self.linear = nn.Linear(width * height + len(CONFIG.cols_annotation), 2)

    def forward(self, features_maps, annotations):
        features_maps = torch.tanh(self.conv(features_maps))
        input = torch.cat(
            (features_maps.reshape(features_maps.size(0), -1), annotations), dim=-1
        )
        probs = torch.softmax(self.linear(input), dim=-1)
        return probs[:, [0]]


class ClassifierNetwork(nn.Module):
    def __init__(self, maps, width, height):
        super(ClassifierNetwork, self).__init__()
        self.conv = nn.Conv2d(maps, 1, kernel_size=1, padding=0, stride=1)
        self.linear = nn.Linear(width * height, 2)

    def forward(self, feature_maps):
        return self.linear(self.conv(feature_maps).reshape(feature_maps.size(0), -1))


class MOEModel(nn.Module):
    def __init__(self):
        super(MOEModel, self).__init__()
        self.cnn = ConvolutionalNetwork()
        self.cnn_classifier = ClassifierNetwork(128, 7, 7)
        self.gate = GatingNetwork(128, 7, 7)
        self.mlp = nn.Sequential(
            nn.Linear(len(CONFIG.cols_annotation), 2),
            nn.Sigmoid(),
            nn.Linear(2, 2),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        images, annotations, batch = x

        features_maps = self.cnn(images)
        features_maps = scatter_mean(features_maps, batch, dim=0)
        y_cnn = self.cnn_classifier(features_maps)

        y_mlp = self.mlp(annotations)[:, [0]]

        pi_cnn = self.gate(features_maps, annotations)
        pi_mlp = 1 - pi_cnn

        return pi_cnn * self.sigmoid(y_cnn) + pi_mlp * self.sigmoid(y_mlp)
