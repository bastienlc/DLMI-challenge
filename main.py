import torch

from src.data import PerPatientDataset
from src.models.cnn import AdditiveModel
from src.train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 4
epochs = 100
load = None

model = AdditiveModel().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

try:
    train(
        model,
        optimizer,
        scheduler,
        PerPatientDataset,
        epochs=epochs,
        batch_size=batch_size,
        load=load,
        device=device,
    )
except KeyboardInterrupt:
    pass
