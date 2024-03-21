import torch

from src.models.cnn import SimpleCNN
from src.train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TRAIN
batch_size = 64
epochs = 100
load = None

model = SimpleCNN().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

try:
    train(
        model,
        optimizer,
        scheduler,
        epochs=epochs,
        batch_size=batch_size,
        load=load,
        device=device,
    )
except KeyboardInterrupt:
    pass
