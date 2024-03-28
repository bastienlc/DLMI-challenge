import torch

from src.data import PerImageDataset
from src.models.baseline import Baseline
from src.train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = Baseline().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.975)
model = train(
    model,
    optimizer,
    scheduler,
    PerImageDataset,
    epochs=200,
    batch_size=128,
    device=device,
    ratio=2,
)
