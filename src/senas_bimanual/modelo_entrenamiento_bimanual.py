import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib

# -------------------------
# CONFIG
# -------------------------
DATA_DIR = "data_dynamic_bimano"
MODEL_PATH = "models/modelo_dinamico_bimanual.pth"
LABELS_PATH = "models/labels_bimano.pkl"

SEQ_LEN = 50
FEATURES = 126
BATCH_SIZE = 16
EPOCHS = 50
LR = 0.0008

os.makedirs("models", exist_ok=True)

# -------------------------
# CARGA DE DATOS
# -------------------------
X, y = [], []
labels = sorted([x for x in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, x))])

label_to_id = {lbl: i for i, lbl in enumerate(labels)}
id_to_label = {i: lbl for lbl, i in label_to_id.items()}
joblib.dump(id_to_label, LABELS_PATH)

for label in labels:
    carpeta = os.path.join(DATA_DIR, label)
    archivos = sorted([f for f in os.listdir(carpeta) if f.endswith(".npy")])

    for f in archivos:
        data = np.load(os.path.join(carpeta, f))

        if data.shape != (SEQ_LEN, FEATURES):
            print(f"Se ignora {f}: dimensiones {data.shape} incorrectas")
            continue

        X.append(data)
        y.append(label_to_id[label])

X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -------------------------
# MODELO TRANSFORMER
# -------------------------
class TransformerBimano(nn.Module):
    def __init__(self, input_dim, num_classes, dim=256, heads=8, layers=4):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.input_fc(x)
        out = self.transformer(x)
        last = out[:, -1, :]
        return self.fc(last)

model = TransformerBimano(FEATURES, len(labels))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

print("Entrenando modelo Transformer bimanual...")

for epoch in range(EPOCHS):
    total_loss = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        out = model(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

torch.save(model.state_dict(), MODEL_PATH)
print("\n✔ Modelo guardado:", MODEL_PATH)
