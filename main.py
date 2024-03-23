import torch

from src.data import CustomLabelsDataset, PerPatientDataset
from src.models.vgg import VGGModel
from src.train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_pretrained = None
model = VGGModel(version="vgg19").to(device)

# Pre-train to classify images on custom labels
model.pre_train()
if load_pretrained is None:
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    model = train(
        model,
        optimizer,
        scheduler,
        CustomLabelsDataset,
        epochs=100,
        batch_size=128,
        device=device,
        ratio=20,
    )
else:
    model.load_state_dict(torch.load(f"{load_pretrained}/model.pt"))

# Fine tune to classify patients on default dataset
model.fine_tune()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
train(
    model,
    optimizer,
    scheduler,
    PerPatientDataset,
    epochs=100,
    batch_size=2,
    device=device,
    ratio=1,
)
