import torch
from sklearn.model_selection import StratifiedKFold

from src.config import CONFIG
from src.data import PerPatientDataset, get_data_loaders
from src.models.moe import MixtureOfExperts
from src.train import train
from src.utils import get_paths_and_labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
folder = StratifiedKFold(n_splits=5, shuffle=True, random_state=CONFIG.SEED)
paths, labels = get_paths_and_labels("train")

for k, (train_index, val_index) in enumerate(folder.split(paths, labels)):
    torch.cuda.empty_cache()
    print(f"Training model {k + 1}")

    model = MixtureOfExperts().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.975)

    model = train(
        model,
        optimizer,
        scheduler,
        PerPatientDataset,
        epochs=100,
        batch_size=8,
        device=device,
        ratio=2,
        loaders=get_data_loaders(
            batch_size=8,
            dataset=PerPatientDataset,
            paths=([paths[i] for i in train_index], [paths[i] for i in val_index]),
        ),
    )
