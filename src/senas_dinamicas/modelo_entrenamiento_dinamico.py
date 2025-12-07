import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib

# ============================================================
# CONFIGURACIÓN
# ============================================================
DATA_DIR = "data_dynamic"               # Carpeta donde el grabador guarda secuencias
MODEL_PATH = "models/modelo_dinamico_transformer.pth"
LABELS_PATH = "models/labels_dinamicos.pkl"

SEQ_LEN = 50                            # frames por secuencia
FEATURES = 63                           # coordenadas mano + landmarks
BATCH_SIZE = 16
EPOCHS = 60
LR = 0.0008

os.makedirs("models", exist_ok=True)

# ============================================================
# CARGA DEL DATASET
# ============================================================
def cargar_dataset():
    X, y = [], []

    # Carpetas = etiquetas
    labels = sorted([
        x for x in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, x))
    ])

    if not labels:
        raise Exception("❌ No hay carpetas dentro de data_dynamic")

    # Guardar mapeo ID → etiqueta
    label_to_id = {lbl: i for i, lbl in enumerate(labels)}
    id_to_label = {i: lbl for lbl, i in label_to_id.items()}
    joblib.dump(id_to_label, LABELS_PATH)

    print("\n📌 ETIQUETAS DETECTADAS:")
    for k, v in id_to_label.items():
        print(f"  {k}: {v}")

    print("\n📂 Cargando secuencias...")

    # Leer secuencias .npy
    for label in labels:
        carpeta = os.path.join(DATA_DIR, label)
        archivos = [f for f in os.listdir(carpeta) if f.endswith(".npy")]

        for archivo in archivos:
            seq = np.load(os.path.join(carpeta, archivo))

            if seq.ndim != 2 or seq.shape[1] != FEATURES:
                print(f"⚠ Ignorada secuencia {archivo} con shape {seq.shape}")
                continue

            # Normalizar longitud
            if seq.shape[0] < SEQ_LEN:
                padding = np.repeat(seq[-1][None, :], SEQ_LEN - seq.shape[0], axis=0)
                seq = np.vstack([seq, padding])
            else:
                seq = seq[:SEQ_LEN]

            X.append(seq)
            y.append(label_to_id[label])

    if len(X) == 0:
        raise Exception("❌ No hay secuencias válidas")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print(f"\n✔ Total secuencias cargadas: {len(X)}")
    return X, y

# ============================================================
# MODELO TRANSFORMER
# ============================================================
class SignTransformer(nn.Module):
    def __init__(self, features, num_classes, seq_len):
        super().__init__()
        embed_dim = 128
        num_heads = 8
        hidden_dim = 256
        num_layers = 3

        self.embedding = nn.Linear(features, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding
        encoded = self.encoder(x)
        pooled = encoded[:, -1, :]
        out = self.cls(pooled)
        return out

# ============================================================
# ENTRENAMIENTO
# ============================================================
def entrenar():
    # Cargar dataset
    X, y = cargar_dataset()

    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    num_classes = len(np.unique(y))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n⚡ Entrenando en: {device}")

    model = SignTransformer(FEATURES, num_classes, SEQ_LEN).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # Bucle de entrenamiento
    for epoch in range(EPOCHS):
        total_loss = 0

        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            opt.zero_grad()
            pred = model(batch_x)
            loss = crit(pred, batch_y)

            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"📘 Epoch {epoch+1}/{EPOCHS}  |  Loss: {total_loss/len(loader):.4f}")

    # Guardar modelo
    torch.save(model.state_dict(), MODEL_PATH)
    print("\n✔ Modelo guardado en:", MODEL_PATH)
    print("✔ Etiquetas guardadas en:", LABELS_PATH)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    entrenar()
