from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .config import CONFIG
from .datasets.custom_labels import CustomLabelsDataset
from .datasets.per_image import PerImageDataset
from .datasets.per_patient import PerPatientDataset, collate
from .utils import get_patients_paths


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


def get_data_loaders(
    test_size=0.2, shuffle=True, batch_size=16, dataset=PerPatientDataset
):
    annotations = preprocess_annotations(pd.read_csv(CONFIG.PATH_TRS_AN))
    patients_paths = get_patients_paths("train")
    train_paths, val_paths = train_test_split(
        patients_paths, test_size=test_size, random_state=CONFIG.SEED
    )

    train_dataset = dataset(train_paths, annotations, split="train")
    val_dataset = dataset(val_paths, annotations, split="val")

    if issubclass(dataset, PerPatientDataset):
        collate_fn = collate
    elif issubclass(dataset, PerImageDataset) or issubclass(
        dataset, CustomLabelsDataset
    ):
        collate_fn = None
    else:
        raise ValueError("Unknown dataset")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def get_test_dataloader(batch_size=16, dataset=PerPatientDataset):
    annotations = preprocess_annotations(pd.read_csv(CONFIG.PATH_TS_AN))
    patients_paths = get_patients_paths("test")
    test_dataset = dataset(patients_paths, annotations, split="test")

    if issubclass(dataset, PerPatientDataset):
        collate_fn = collate
    elif issubclass(dataset, PerImageDataset) or issubclass(
        dataset, CustomLabelsDataset
    ):
        collate_fn = None
    else:
        raise ValueError("Unknown dataset")

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    return test_loader
