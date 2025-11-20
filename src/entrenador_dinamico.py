import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib

# -------------------------
# CONFIG
# -------------------------
DATA_DIR = "data_dynamic"
MODEL_PATH = "models/modelo_dinamico.pth"
LABELS_PATH = "models/labels_dinamicos.pkl"

SEQ_LEN = 20
FEATURES = 63
BATCH_SIZE = 16
EPOCHS = 50
LR = 0.001

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

    frames = sorted([f for f in os.listdir(carpeta) if f.endswith(".npy")])

    if len(frames) == 0:
        print(f"⚠ Señal '{label}' ignorada (no tiene frames).")
        continue

    seq = []

    for f in frames:
        data = np.load(os.path.join(carpeta, f))

        # ------- ARREGLAR DIMENSIONES INCORRECTAS -------
        data = np.array(data).reshape(-1)      # <— SIEMPRE queda (63,)
        if data.shape[0] != FEATURES:
            print(f"⚠ Frame ignorado por tamaño incorrecto: {f} en {label}")
            continue

        seq.append(data)

    seq = np.array(seq)

    # ---------------- AJUSTAR LONGITUD A SEQ_LEN ----------------
    if len(seq) == 0:
        print(f"⚠ Señal '{label}' ignorada (frames inválidos).")
        continue

    if len(seq) < SEQ_LEN:
        padding = np.zeros((SEQ_LEN - len(seq), FEATURES))
        seq = np.vstack([seq, padding])
    else:
        seq = seq[:SEQ_LEN]

    X.append(seq)
    y.append(label_to_id[label])

# -------------------------
# VALIDACIÓN FINAL
# -------------------------
if len(X) == 0:
    raise Exception("❌ ERROR: Ninguna carpeta contiene secuencias válidas.")

X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -------------------------
# MODELO LSTM
# -------------------------
class ModeloLSTM(nn.Module):
    def __init__(self, input_size, hidden, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

model = ModeloLSTM(FEATURES, 128, len(labels))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -------------------------
# TRAINING
# -------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

print("Entrenando modelo dinámico en:", device)

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
print("\n✔ Modelo dinámico guardado:", MODEL_PATH)
