import os
import random

import numpy as np
import torch


class CONFIG:
    PATH_INPUT = "./data"
    PATH_TRS = os.path.join(PATH_INPUT, "trainset")
    PATH_TRS_AN = os.path.join(PATH_TRS, "trainset_true.csv")
    PATH_TS = os.path.join(PATH_INPUT, "testset")
    PATH_TS_AN = os.path.join(PATH_TS, "testset_data.csv")
    PATH_AN = os.path.join(PATH_INPUT, "clinical_annotation.csv")
    PATH_SUB = os.path.join(PATH_INPUT, "sample_submission.csv")

    cols_annotation = ["GENDER", "LYMPH_COUNT", "AGE"]
    col_label = "LABEL"

    SEED = 42


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


seed_everything(
    CONFIG.SEED
)  # Reproducibility; still make sure to pass CONFIG.SEED to split functions
