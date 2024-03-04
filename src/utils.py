import os
import time
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class TrainLogger:
    """Save checkpoints, model and training parameters, log training and validation loss and accuracy, all in one class."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        parameters: Dict,
        dir="runs",
        load=None,
    ):
        self.dir = dir

        self.train_loss = 0
        self.val_loss = 0
        self.val_accuracy = 0
        self.best_accuracy = 0
        self.last_epoch = 0

        if load is None:
            self.save_dir = f"{self.dir}/{time.strftime('%d-%m_%H:%M:%S')}"

            if not os.path.exists(self.dir):
                os.makedirs(self.dir)
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

            with open(f"{self.save_dir}/model.txt", "w") as file:
                file.write(str(model))

            with open(f"{self.save_dir}/parameters.txt", "w") as file:
                file.write(str(parameters))

            self.save(model, optimizer, 0)

        else:
            self.save_dir = load

            with open(f"{self.save_dir}/model.txt", "r") as file:
                if file.readline() != str(model):
                    raise ValueError("Model does not match the one saved")

            with open(f"{self.save_dir}/parameters.txt", "r") as file:
                saved_parameters = file.readline()
                if saved_parameters != str(parameters):
                    print("Warning: training resumed with different parameters.")
                    print("Saved:", saved_parameters)
                    print("Current:", parameters)

            self.best_accuracy = 0
            with open(f"{self.save_dir}/val_accuracy.csv", "r") as file:
                for line in file:
                    epoch, accuracy = line.split(",")
                    self.best_accuracy = max(self.best_accuracy, float(accuracy))

                self.last_epoch = int(epoch)

        self.summary_writer = SummaryWriter(self.save_dir)

    def log(
        self,
        epoch,
        train_loss=None,
        val_loss=None,
        val_accuracy=None,
        additional_metrics=None,
    ):
        if train_loss is not None:
            self.summary_writer.add_scalar("loss/train", train_loss, epoch)
            with open(f"{self.save_dir}/train_loss.csv", "a") as file:
                file.write(f"{epoch},{train_loss}\n")
            self.train_loss = train_loss

        if val_loss is not None:
            self.summary_writer.add_scalar("loss/val", val_loss, epoch)
            with open(f"{self.save_dir}/val_loss.csv", "a") as file:
                file.write(f"{epoch},{val_loss}\n")
            self.val_loss = val_loss

        if val_accuracy is not None:
            self.summary_writer.add_scalar("accuracy/val", val_accuracy, epoch)
            with open(f"{self.save_dir}/val_accuracy.csv", "a") as file:
                file.write(f"{epoch},{val_accuracy}\n")
            self.val_accuracy = val_accuracy

            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy

        if additional_metrics is not None:
            for name, value in additional_metrics.items():
                self.summary_writer.add_scalar(name, value, epoch)
                with open(f"{self.save_dir}/{name}.csv", "a") as file:
                    file.write(f"{epoch},{value}\n")

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        val_accuracy: float,
    ):
        if val_accuracy >= self.best_accuracy:
            torch.save(model.state_dict(), f"{self.save_dir}/model.pt")
            torch.save(optimizer.state_dict(), f"{self.save_dir}/optimizer.pt")
            self.best_accuracy = val_accuracy

    def load(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        model.load_state_dict(torch.load(f"{self.save_dir}/model.pt"))
        optimizer.load_state_dict(torch.load(f"{self.save_dir}/optimizer.pt"))
        return model, optimizer

    def print(self, epoch):
        print(
            f"Epoch {epoch}: train_loss={self.train_loss:.2E}, val_loss={self.val_loss:.2E}, val_accuracy={self.val_accuracy:.2f}",
        )

    def __del__(self):
        self.summary_writer.flush()
        self.summary_writer.close()


def predict(model, loader):
    model.eval()
    with torch.no_grad():
        y_pred = []
        for data in tqdm(loader):
            output = model(data[0])
            y_pred.append(output.argmax(1).cpu().numpy())
    return np.concatenate(y_pred)


def save(y_pred, index, file_name="solution.csv"):
    y_pred = pd.DataFrame(y_pred, columns=["LABEL"])
    y_pred["ID"] = index
    y_pred.to_csv(
        file_name,
        index=False,
        columns=["ID", "LABEL"],
        header=["ID", "LABEL"],
    )
