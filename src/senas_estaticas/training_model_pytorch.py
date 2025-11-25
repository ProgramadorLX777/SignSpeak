import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib

# --- CONFIGURACIÓN ---
DATA_DIR = "datos"
MODEL_PATH = "models/modelo_pytorch.pth"
LABELS_PATH = "models/labels.pkl"

INPUT_SIZE = 126   # ⬅⬅⬅ AHORA CORRECTO
HIDDEN1 = 128
HIDDEN2 = 64
EPOCHS = 40
BATCH_SIZE = 32
LR = 0.001

# --- CARGAR DATASET ---
X = []
y = []
labels = []

for archivo in sorted(os.listdir(DATA_DIR)):   # orden estable
    if archivo.endswith(".csv"):
        etiqueta = archivo.replace(".csv", "")
        labels.append(etiqueta)

        with open(f"{DATA_DIR}/{archivo}", "r") as f:
            reader = csv.reader(f)
            for fila in reader:
                fila_float = [float(x) for x in fila]

                # Asegurarse que todas las filas tengan 126 valores EXACTOS
                if len(fila_float) < 126:
                    fila_float.extend([0.0] * (126 - len(fila_float)))
                elif len(fila_float) > 126:
                    fila_float = fila_float[:126]

                X.append(fila_float)
                y.append(etiqueta)

# Mapear etiquetas a números
label_to_id = {label: i for i, label in enumerate(labels)}
id_to_label = {i: label for label, i in label_to_id.items()}
joblib.dump(id_to_label, LABELS_PATH)

y_num = [label_to_id[label] for label in y]

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y_num, dtype=torch.long)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- DEFINIR EL MODELO ---
class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, HIDDEN1),
            nn.ReLU(),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.ReLU(),
            nn.Linear(HIDDEN2, num_classes)
        )

    def forward(self, x):
        return self.model(x)

model = MLP(INPUT_SIZE, len(labels))

# Usar GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Entrenando modelo mediante:", device)
model.to(device)

# --- ENTRENAMIENTO ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    running_loss = 0.0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss:.4f}")

torch.save(model.state_dict(), MODEL_PATH)
print("Modelo guardado en: ", MODEL_PATH)
