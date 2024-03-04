import numpy as np
import torch

from src.data import get_test_dataloader
from src.models.cnn import SimpleCNN
from src.train import train
from src.utils import save

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TRAIN
batch_size = 64
epochs = 100
load = None

model = SimpleCNN().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

try:
    model = train(
        model,
        optimizer,
        scheduler,
        epochs=epochs,
        batch_size=batch_size,
        load=load,
    )
except KeyboardInterrupt:
    pass

# TEST
model.eval()
test_loader = get_test_dataloader(batch_size=batch_size)

with torch.no_grad():
    y_pred = {}
    for data in test_loader:
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
