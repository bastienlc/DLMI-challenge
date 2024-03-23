import torch
import torch.nn as nn
from torch_scatter import scatter_mean
from torchvision import models

from ..config import CONFIG


class VGG(nn.Module):
    def __init__(self, dropout=0.05, version="vgg19"):
        super(VGG, self).__init__()
        if version == "vgg16":
            self.vgg = models.vgg16(weights="DEFAULT")
        else:
            self.vgg = models.vgg19(weights="DEFAULT")
        self.vgg.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 2),
        )

    def forward(self, x):
        return self.vgg(x)


class VGGModel(nn.Module):
    def __init__(self, dropout=0.05, version="vgg19"):
        super(VGGModel, self).__init__()
        self.image_model = VGG(dropout=dropout, version=version)
        self.linear_model = nn.Sequential(
            nn.Linear(4096 + len(CONFIG.cols_annotation), 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, 2),
        )
        self._pre_train = True

    def pre_train_step(self, x):
        images, _, _ = x
        return self.image_model(images)

    def fine_tune_step(self, x):
        images, annotations, batch = x
        y = self.image_model(images)
        y = scatter_mean(y, batch, dim=0)
        y = torch.cat((y, annotations), dim=1)
        return self.linear_model(y)

    def forward(self, x):
        if self._pre_train:
            return self.pre_train_step(x)
        else:
            return self.fine_tune_step(x)

    def pre_train(self):
        self._pre_train = True

    def fine_tune(self):
        self._pre_train = False
        self.image_model.vgg.classifier[-1] = nn.Identity()
        for param in self.image_model.vgg.parameters():
            param.requires_grad = False
