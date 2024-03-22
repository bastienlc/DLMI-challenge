import os
from typing import List

import matplotlib.image as mpimg
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset
from torchvision.transforms import v2
from tqdm import tqdm

from ..config import CONFIG
from ..utils import get_patient_id_from_path, get_patient_images_paths


class PerImageDataset(Dataset):
    def __init__(
        self,
        patients_paths: List[str],
        df,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        split: str = "train",
        name="per_image",
        image_crop_size=150,
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
                for idx in range(self.num_images.sum())
            ]
        )

    def process(self):
        idx = 0
        for k in tqdm(range(len(self.patients_paths)), desc="Preprocessing data"):
            patient_id = get_patient_id_from_path(self.patients_paths[k])
            number_id = int(patient_id[1:])
            annotations = self.df.loc[(self.df["ID"] == patient_id)][
                CONFIG.cols_annotation
            ]
            label = self.df.loc[(self.df["ID"] == patient_id)][CONFIG.col_label].values[
                0
            ]
            annotations = np.array(annotations, dtype=np.float32).squeeze()
            patient_images_paths = get_patient_images_paths(self.patients_paths[k])

            for _, img_path in enumerate(patient_images_paths):
                img = mpimg.imread(img_path)
                img = np.moveaxis(img, -1, 0)

                torch.save(
                    (
                        self.transform(torch.tensor(img, device="cpu")),
                        torch.tensor(annotations, device="cpu"),
                        torch.tensor(number_id, device="cpu"),
                        torch.tensor(label, device="cpu"),
                    ),
                    os.path.join(self.processed_dir, f"{idx}.pt"),
                )
                idx += 1

    def __len__(self):
        return self.num_images.sum()

    def __getitem__(self, idx):
        images, annotations, patient_id, label = torch.load(
            os.path.join(self.processed_dir, f"{idx}.pt")
        )

        images = self.online_transform(images)

        return images, annotations, patient_id, label
