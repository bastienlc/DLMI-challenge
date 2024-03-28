from collections import defaultdict

import numpy as np
import torch

from src.data import get_test_dataloader
from src.datasets.per_image import PerImageDataset
from src.datasets.per_patient import PerPatientDataset
from src.models.moe import MOEModel
from src.utils import (
    get_patient_id_from_patient_path,
    get_patients_paths,
    save,
    to_device,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(model, dataset, device):
    model.eval()
    loader = get_test_dataloader(batch_size=16, dataset=dataset)

    if issubclass(dataset, PerPatientDataset):
        per_patient = True
    elif issubclass(dataset, PerImageDataset):
        per_patient = False
    else:
        raise ValueError("Unknown dataset")

    with torch.no_grad():
        if per_patient:
            predictions = []
            patients_paths = get_patients_paths("test")
            index = [get_patient_id_from_patient_path(path) for path in patients_paths]
            for data in loader:
                data = to_device(data, device)
                output = model(data[0:-1])
                predictions.append(torch.argmax(output, dim=1).cpu().numpy())
            return np.concatenate(predictions), index
        else:
            predictions = defaultdict(list)
            for data in loader:
                data = to_device(data, device)
                output = model(data[0:-1])
                for idx, patient_id in enumerate(data[2]):
                    patient_id = patient_id.item()
                    predictions[patient_id].append(torch.argmax(output[idx]).item())

            return np.array(
                [
                    round(np.mean(predictions[patient_id]))
                    for patient_id in predictions.keys()
                ]
            ), [f"P{patient_id}" for patient_id in predictions.keys()]


if __name__ == "__main__":
    dataset = PerPatientDataset
    name = "mixture_of_experts"
    model = MOEModel().to(device)

    model.load_state_dict(torch.load(f"runs/{name}/model.pt"))
    model.eval()
    predictions, index = predict(model, dataset, device)
    save(predictions, index, file_name=f"runs/{name}/solution.csv")
