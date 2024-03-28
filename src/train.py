from typing import Union

import torch
import torch.nn as nn
from tqdm import tqdm

from .data import get_data_loaders
from .utils import TrainLogger, to_device


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    dataset: torch.utils.data.Dataset,
    epochs: int = 100,
    batch_size: int = 32,
    load: Union[str, None] = None,
    ratio: float = 1,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    loaders: Union[None, torch.utils.data.DataLoader] = None,
):
    """
    Main training loop. This function does everything from loading the data to saving the model. If a loader is passed, it will use that one (for cross-validation).
    """
    logger = TrainLogger(
        model, optimizer, {"epochs": epochs, "batch_size": batch_size}, load=load
    )

    model, optimizer = logger.load(model, optimizer)

    loss_function = torch.nn.CrossEntropyLoss(
        weight=torch.tensor([1, ratio], dtype=torch.float32).to(device),
        reduction="mean",
    )

    if loaders is None:
        train_loader, val_loader = get_data_loaders(
            batch_size=batch_size, dataset=dataset
        )
    else:
        train_loader, val_loader = loaders
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)

    for epoch in range(logger.last_epoch + 1, epochs + 1):
        model.train()
        (
            train_loss,
            true_positives,
            false_positives,
            true_negatives,
            false_negatives,
        ) = (0, 0, 0, 0, 0)

        # TRAIN
        progress_bar = tqdm(train_loader, leave=False)
        for data in progress_bar:
            data = to_device(data, device)
            target = data[-1]
            output = model(data[0:-1])
            loss = loss_function(output, target)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({"loss": f"{loss/batch_size:.2E}"})

            prediction = torch.argmax(output, dim=1)
            true_positives += ((prediction == target) & (target == 1)).sum().item()
            false_positives += ((prediction != target) & (target == 0)).sum().item()
            true_negatives += ((prediction == target) & (target == 0)).sum().item()
            false_negatives += ((prediction != target) & (target == 1)).sum().item()

        scheduler.step()

        train_balanced_accuracy = (
            (true_positives / (true_positives + false_negatives))
            + (true_negatives / (true_negatives + false_positives))
        ) / 2

        logger.log(
            epoch,
            train_loss=train_loss / n_train,
            additional_metrics={
                "train_true_positives_rate": true_positives
                / (true_positives + false_negatives),
                "train_true_negatives_rate": true_negatives
                / (true_negatives + false_positives),
                "train_accuracy": train_balanced_accuracy,
            },
        )

        # EVAL
        model.eval()
        with torch.no_grad():
            (
                val_loss,
                true_positives,
                false_positives,
                true_negatives,
                false_negatives,
            ) = (0, 0, 0, 0, 0)
            for data in val_loader:
                data = to_device(data, device)
                output = model(data[0:-1])
                target = data[-1]
                val_loss += loss_function(output, target).item()
                prediction = torch.argmax(output, dim=1)
                true_positives += ((prediction == target) & (target == 1)).sum().item()
                false_positives += ((prediction != target) & (target == 0)).sum().item()
                true_negatives += ((prediction == target) & (target == 0)).sum().item()
                false_negatives += ((prediction != target) & (target == 1)).sum().item()

        val_balanced_accuracy = (
            (true_positives / (true_positives + false_negatives))
            + (true_negatives / (true_negatives + false_positives))
        ) / 2

        logger.log(
            epoch,
            val_loss=val_loss / n_val,
            val_accuracy=val_balanced_accuracy,
            additional_metrics={
                "val_true_positives_rate": true_positives
                / (true_positives + false_negatives),
                "val_true_negatives_rate": true_negatives
                / (true_negatives + false_positives),
                "learning_rate": optimizer.param_groups[0]["lr"],
            },
        )
        logger.save(model, optimizer, val_accuracy=val_balanced_accuracy)
        logger.print(
            epoch,
            special={"train_accuracy": train_balanced_accuracy},
        )

    return model
