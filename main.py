import torch

from src.data import PerPatientDataset
from src.models.moe import MOEModel
from src.train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


load = None
model = MOEModel().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.975)
model = train(
    model,
    optimizer,
    scheduler,
    PerPatientDataset,
    epochs=100,
    batch_size=4,
    device=device,
    load=load,
)
