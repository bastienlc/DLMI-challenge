import os
from typing import List

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch.utils.data import Dataset
from torchvision.transforms import v2
from tqdm import tqdm

from ..config import CONFIG
from ..utils import (
    RandomDiscreteRotation,
    get_patient_id_from_image_path,
    get_patient_id_from_patient_path,
    get_patient_images_paths,
)


class CustomLabelsDataset(Dataset):
    def __init__(
        self,
        patients_paths: List[str],
        df: pd.DataFrame,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        split: str = "train",
        name="manual",
        image_crop_size=112,
    ):
        self.patients_paths = patients_paths
        self.df = df
        self.device = device
        self.split = split
        self.name = name

        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                v2.CenterCrop(image_crop_size),
            ]
        )
        if self.split == "train":
            self.online_transform = v2.Compose(
                [
                    v2.RandomHorizontalFlip(),
                    v2.RandomVerticalFlip(),
                    RandomDiscreteRotation([0, 90, 180, 270]),
                ]
            )
        else:
            self.online_transform = v2.Compose([v2.Identity()])

        self.paths = self.get_paths(
            pd.read_csv(os.path.join(CONFIG.PATH_INPUT, "custom_positives.csv"))
        )

        self.processed_dir = os.path.join("./data", "processed", self.name, self.split)
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        if self.should_process():
            self.process()

    def get_paths(self, positives: pd.DataFrame):
        paths = []
        self.positive_images_paths = [
            os.path.join(CONFIG.PATH_TRS, patient_id, image_name)
            for patient_id, image_name in zip(positives["patient"], positives["image"])
        ]
        for k in range(len(self.patients_paths)):
            patient_id = get_patient_id_from_patient_path(self.patients_paths[k])
            label = self.df.loc[(self.df["ID"] == patient_id)][CONFIG.col_label].values[
                0
            ]
            patient_paths = get_patient_images_paths(self.patients_paths[k])
            for path in patient_paths:
                if label == 0 or path in self.positive_images_paths:
                    paths.append(path)
        return paths

    def should_process(self):
        return not all(
            [
                os.path.exists(os.path.join(self.processed_dir, f"{idx}.pt"))
                for idx in range(len(self.paths))
            ]
        )

    def process(self):
        for idx, path in enumerate(tqdm(self.paths, desc="Preprocessing data")):
            patient_id = get_patient_id_from_image_path(path)
            number_id = int(patient_id[1:])
            annotations = self.df.loc[(self.df["ID"] == patient_id)][
                CONFIG.cols_annotation
            ]
            label = self.df.loc[(self.df["ID"] == patient_id)][CONFIG.col_label].values[
                0
            ]
            annotations = np.array(annotations, dtype=np.float32).squeeze()

            img = mpimg.imread(path)
            img = np.moveaxis(img, -1, 0)
            img = img.astype(np.float32)

            torch.save(
                (
                    self.transform(torch.tensor(img, device="cpu")),
                    torch.tensor(annotations, device="cpu"),
                    torch.tensor(number_id, device="cpu"),
                    torch.tensor(label, device="cpu"),
                ),
                os.path.join(self.processed_dir, f"{idx}.pt"),
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image, annotations, patient_id, label = torch.load(
            os.path.join(self.processed_dir, f"{idx}.pt")
        )
        image = self.online_transform(image)

        return image, annotations, patient_id, label
