import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max

from ..config import CONFIG


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 34 * 34 + len(CONFIG.cols_annotation), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        image, annotations, _ = x
        image = self.pool(F.relu(self.conv1(image)))
        image = self.pool(F.relu(self.conv2(image)))
        image = image.view(-1, 16 * 34 * 34)
        x = torch.cat((image, annotations), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AdditiveModel(nn.Module):
    def __init__(self, dropout=0.05):
        super(AdditiveModel, self).__init__()
        self.image_model = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.linear_model = nn.Sequential(
            nn.Linear(18496 + len(CONFIG.cols_annotation), 120),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 2),
        )

    def forward(self, x):
        images, annotations, batch = x
        y = self.image_model(images)
        y, _ = scatter_max(y, batch, dim=0)
        y = torch.cat((y.view(annotations.shape[0], -1), annotations), dim=1)

        return self.linear_model(y)
