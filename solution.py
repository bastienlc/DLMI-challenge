import torch

from src.data import get_test_dataloader
from src.models.moe import MOEModel
from src.utils import (
    get_patient_id_from_patient_path,
    get_patients_paths,
    predict,
    save,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

name = "segment"

model = MOEModel().to(device)
model.load_state_dict(torch.load(f"runs/{name}/model.pt"))
model.eval()
test_loader = get_test_dataloader(batch_size=256)

predictions = predict(model, test_loader, device)

patients_paths = get_patients_paths("test")
index = [get_patient_id_from_patient_path(path) for path in patients_paths]

save(predictions, index)
