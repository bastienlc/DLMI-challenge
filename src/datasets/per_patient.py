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
from ..utils import get_patient_id_from_path, get_patient_images_paths


class PerPatientDataset(Dataset):
    def __init__(
        self,
        patients_paths: List[str],
        df: pd.DataFrame,
        split: str = "train",
        name: str = "per_patient",
        image_crop_size=150,
    ):
        self.patients_paths = patients_paths
        self.df = df
        self.split = split
        self.name = name

        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        if self.split == "train":
            self.online_transform = v2.Compose(
                [
                    v2.RandomHorizontalFlip(),
                    v2.RandomVerticalFlip(),
                    v2.RandomRotation(degrees=180),
                    v2.CenterCrop(image_crop_size),
                ]
            )
        else:
            self.online_transform = v2.Compose([v2.CenterCrop(image_crop_size)])

        self.num_images = np.zeros(len(self.patients_paths), dtype=int)
        for idx in range(len(self.patients_paths)):
            self.num_images[idx] = len(
                get_patient_images_paths(self.patients_paths[idx])
            )

        self.processed_dir = os.path.join("./data", "processed", self.name, self.split)
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        if self.should_process():
            self.process()

    def should_process(self):
        return not all(
            [
                os.path.exists(os.path.join(self.processed_dir, f"{idx}.pt"))
                for idx in range(len(self.patients_paths))
            ]
        )

    def process(self):
        for idx in tqdm(range(len(self.patients_paths)), desc="Preprocessing data"):
            patient_id = get_patient_id_from_path(self.patients_paths[idx])
            annotations = self.df.loc[(self.df["ID"] == patient_id)][
                CONFIG.cols_annotation
            ]
            label = self.df.loc[(self.df["ID"] == patient_id)][CONFIG.col_label].values[
                0
            ]
            annotations = np.array(annotations, dtype=np.float32).squeeze()

            patient_images_paths = get_patient_images_paths(self.patients_paths[idx])
            patient_images = np.zeros((len(patient_images_paths), 3, 224, 224))

            for k, img_path in enumerate(patient_images_paths):
                img = mpimg.imread(img_path)
                img = np.moveaxis(img, -1, 0)
                patient_images[k] = img

            torch.save(
                (
                    self.transform(torch.tensor(patient_images, device="cpu")),
                    torch.tensor(annotations, device="cpu"),
                    torch.tensor(label, device="cpu"),
                ),
                os.path.join(self.processed_dir, f"{idx}.pt"),
            )

    def __len__(self):
        return len(self.patients_paths)

    def __getitem__(self, idx):
        images, annotations, label = torch.load(
            os.path.join(self.processed_dir, f"{idx}.pt")
        )

        images = self.online_transform(images)

        return (
            images,
            annotations,
            torch.zeros(images.shape[0], dtype=torch.long),
            label,
        )


def batch_from_data_list(data_list: List[torch.LongTensor]):
    cumulated = -1
    result = []
    for batch in data_list:
        previous = None
        for i in batch:
            if i != previous:
                cumulated += 1
                previous = i
            result.append(cumulated)

    return torch.tensor(result, dtype=torch.long)


def collate(data):
    images, annotations, batch, labels = zip(*data)
    return (
        torch.concatenate(images),
        torch.stack(annotations),
        batch_from_data_list(batch),
        torch.stack(labels),
    )
