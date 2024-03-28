import glob
import os
import random
import time
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from skimage.morphology import (
    area_closing,
    area_opening,
    binary_dilation,
    convex_hull_image,
)
from torch.utils.tensorboard import SummaryWriter

from .config import CONFIG


class TrainLogger:
    """Save checkpoints, model and training parameters, log training and validation loss and accuracy, all in one class."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        parameters: Dict,
        dir="runs",
        load=None,
        raise_on_mismatch=False,
    ):
        self.dir = dir
        self.raise_on_mismatch = raise_on_mismatch

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
                saved_model = file.read()
                if saved_model != str(model):
                    if self.raise_on_mismatch:
                        raise AssertionError(
                            f"Model does not match the one saved, expected {saved_model} but got {str(model)}"
                        )
                    else:
                        print("Warning: training resumed with different model.")
                        print("Saved:", saved_model)
                        print("Current:", str(model))

            with open(f"{self.save_dir}/parameters.txt", "r") as file:
                saved_parameters = file.read()
                if saved_parameters != str(parameters):
                    if self.raise_on_mismatch:
                        raise AssertionError(
                            f"Parameters do not match the ones saved, expected {saved_parameters} but got {str(parameters)}"
                        )
                    else:
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

    def print(self, epoch, special):
        message = f"Epoch {epoch}: train_loss={self.train_loss:.2E}, val_loss={self.val_loss:.2E}, val_accuracy={self.val_accuracy:.2f}"
        for key, value in special.items():
            message += f", {key}={value:.2f}"
        print(message)

    def __del__(self):
        self.summary_writer.flush()
        self.summary_writer.close()


def to_device(data, device):
    images, annotations, batch, labels = data
    return (
        images.to(device),
        annotations.to(device),
        batch.to(device),
        labels.to(device),
    )


def save(y_pred, index, file_name="solution.csv"):
    y_pred = pd.DataFrame(y_pred, columns=["Predicted"])
    y_pred["Id"] = index
    y_pred.to_csv(
        file_name,
        index=False,
        columns=["Id", "Predicted"],
        header=["Id", "Predicted"],
    )


def get_patient_id_from_patient_path(path):
    """
    return patient id from patient path
    """
    return path.split("/")[-1]


def get_patient_id_from_image_path(path):
    """
    return patient id from patient path
    """
    return path.split("/")[-2]


def get_patients_paths(fold: str):
    """
    return paths of training or testing patients
    """
    if fold == "train":
        paths = glob.glob(os.path.join(CONFIG.PATH_TRS, "P*"))
    elif fold == "test":
        paths = glob.glob(os.path.join(CONFIG.PATH_TS, "P*"))

    return paths


def get_patient_images_paths(patient_path: str):
    """
    return paths of images for a given patient
    """
    paths = glob.glob(os.path.join(patient_path, "*.jpg"))

    return paths


def get_paths_and_labels(fold: str):
    """
    Return paths of patients with their corresponding labels. Used for stratified cross_validation.
    """
    annotations = pd.read_csv(CONFIG.PATH_TRS_AN)
    annotations.sort_values("ID", inplace=True)

    paths = get_patients_paths(fold)
    labels = annotations[
        annotations["ID"].isin(
            [get_patient_id_from_patient_path(path) for path in paths]
        )
    ][CONFIG.col_label].values

    return paths, labels


class RandomDiscreteRotation:
    def __init__(self, angles: List[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


def segment_lymphocyt(image: np.ndarray) -> np.ndarray:
    saturation = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2HSV)[:, :, 1]

    _, label, center = cv2.kmeans(
        saturation.reshape((-1)),
        2,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1),
        1,
        cv2.KMEANS_PP_CENTERS,
    )

    # segmentation
    segmented = center[label.flatten()]
    segmented = (segmented == np.max(segmented)).reshape(image.shape[:2])

    # postprocessing
    segmented = area_closing(segmented, area_threshold=50)
    segmented = binary_dilation(segmented, footprint=np.ones((10, 10)))
    segmented = area_opening(segmented, area_threshold=500)
    segmented = convex_hull_image(segmented)

    # add background
    background = np.zeros_like(image)
    background[:, :] = [255, 229, 202]
    segmented = image * segmented[:, :, None] + background * (1 - segmented[:, :, None])

    return segmented
