import glob
import os
from datetime import datetime

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm import tqdm

from .config import CONFIG


def DOB_to_age(date_str):
    if "/" in date_str:
        bday = datetime.strptime(date_str, "%m/%d/%Y")
    elif "-" in date_str:
        bday = datetime.strptime(date_str, "%d-%m-%Y")
    current_date = datetime.now()
    age_in_days = (current_date - bday).days
    return age_in_days / 365.25


def f_to_F(gender):
    if "f" in gender:
        gender = "F"
    return gender


def preprocess_annotations(df, normalize=True):
    """
    preprocessing steps on annotation csv
    """

    # create age
    df["AGE"] = df["DOB"].apply(lambda x: DOB_to_age(x))
    df.drop(columns=["DOB"], inplace=True)

    # uniformize and encoder gender
    df["GENDER"] = df["GENDER"].apply(lambda x: f_to_F(x))
    df["GENDER"] = df["GENDER"].apply(lambda x: 1 if x == "F" else 0)

    # min-max scaling with precomputed values
    if normalize:
        df["AGE"] = (df["AGE"] - 24.97) / (103.15 - 24.97)
        df["LYMPH_COUNT"] = (df["LYMPH_COUNT"] - 2.28) / (295.0 - 2.28)

    return df


def get_file_paths(fold):
    """
    return paths of training or testing images
    """
    if fold == "train":
        l_files = glob.glob(os.path.join(CONFIG.PATH_TRS, "*/*.jpg"))
    elif fold == "test":
        l_files = glob.glob(os.path.join(CONFIG.PATH_TS, "*/*.jpg"))

    return l_files


def get_id_from_path(path):
    patient_id = path.split("/")[-2]
    img_id = path.split("/")[-1].split(".")[-2]
    return patient_id, img_id


class LymphocytosisDataset(Dataset):
    def __init__(
        self,
        l_path,
        df,
        split: str = "train",
    ):
        self.l_path = l_path
        self.df = df
        self.split = split
        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.online_transform = v2.Compose(
            [
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.RandomRotation(degrees=180),
                v2.RandomResizedCrop(
                    size=224, scale=(0.8, 1.0), antialias=True
                ),  # images will be resized to 224x224 which is their original size
            ]
        )

        self.processed_dir = os.path.join("./data/processed", self.split)
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        if self.should_process():
            self.process()

    def should_process(self):
        return not all(
            [
                os.path.exists(os.path.join(self.processed_dir, f"{idx}.pt"))
                for idx in range(len(self.l_path))
            ]
        )

    def process(self):
        for idx in tqdm(range(len(self.l_path)), desc="Preprocessing data"):
            patient_id, _ = get_id_from_path(self.l_path[idx])
            img = mpimg.imread(self.l_path[idx])
            img = np.moveaxis(img, -1, 0)
            annotations = self.df.loc[(self.df["ID"] == patient_id)][
                CONFIG.cols_annotation
            ]
            label = self.df.loc[(self.df["ID"] == patient_id)][CONFIG.col_label].values[
                0
            ]
            annotations = np.array(annotations, dtype=np.float32).squeeze()

            torch.save(
                (
                    (
                        self.transform(torch.tensor(img, device="cpu")),
                        torch.tensor(annotations, device="cpu"),
                    ),
                    torch.tensor(label, device="cpu"),
                    patient_id,
                ),
                os.path.join(self.processed_dir, f"{idx}.pt"),
            )

    def __len__(self):
        return len(self.l_path)

    def __getitem__(self, idx):
        if self.split == "train":
            (image, annotations), label, patient_id = torch.load(
                os.path.join(self.processed_dir, f"{idx}.pt")
            )
            return (self.online_transform(image), annotations), label, patient_id
        else:
            return torch.load(os.path.join(self.processed_dir, f"{idx}.pt"))


def get_data_loaders(test_size=0.2, shuffle=True, batch_size=16):
    annotations = preprocess_annotations(pd.read_csv(CONFIG.PATH_TRS_AN))
    images_paths = get_file_paths("train")
    train_images_paths, val_images_paths = train_test_split(
        images_paths, test_size=test_size
    )

    train_dataset = LymphocytosisDataset(train_images_paths, annotations, split="train")
    val_dataset = LymphocytosisDataset(val_images_paths, annotations, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_dataloader(batch_size=16):
    annotations = preprocess_annotations(pd.read_csv(CONFIG.PATH_TS_AN))
    images_paths = get_file_paths("test")
    test_dataset = LymphocytosisDataset(images_paths, annotations, split="test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


def to_device(data, device):
    return (
        (
            data[0][0].to(device),
            data[0][1].to(device),
        ),
        data[1].to(device),
        data[2],
    )
