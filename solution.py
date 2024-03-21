import numpy as np
import torch

from src.data import get_test_dataloader, to_device
from src.models.cnn import SimpleCNN
from src.utils import save

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

name = "randomize"

model = SimpleCNN().to(device)
model.load_state_dict(torch.load(f"runs/{name}/model.pt"))
model.eval()
test_loader = get_test_dataloader(batch_size=256)

with torch.no_grad():
    y_pred = {}
    for data in test_loader:
        data = to_device(data, device)
        output = model(data[0])
        prediction = output.argmax(1).cpu().numpy()
        for i, id in enumerate(data[2]):
            try:
                y_pred[id].append(prediction[i])
            except KeyError:
                y_pred[id] = [prediction[i]]

# Take the most frequent prediction for each patient
for id in y_pred:
    y_pred[id] = np.bincount(y_pred[id]).argmax()

index = [id for id in sorted(y_pred.keys())]
y_pred = np.array([y_pred[id] for id in sorted(y_pred.keys())])

save(y_pred, index)
