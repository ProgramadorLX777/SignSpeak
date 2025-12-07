import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import joblib
import time

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "models/modelo_dinamico_transformer.pth"
LABELS_PATH = "models/labels_dinamicos.pkl"

SEQ_LEN = 50
FEATURES = 63

WINDOW_STABLE = 8          # Frames necesarios para aceptar un cambio
PERDIDA_TOLERANCIA = 3
FPS_COOLDOWN = 0.15        # Pequeño delay anti-spam

# ============================================================
# CARGAR ETIQUETAS
# ============================================================
id_to_label = joblib.load(LABELS_PATH)

# ============================================================
# MODELO
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
# CARGAR MODELO
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SignTransformer(FEATURES, len(id_to_label), SEQ_LEN).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ============================================================
# MEDIAPIPE
# ============================================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# ============================================================
# FUNCIONES
# ============================================================
def extraer_landmarks(results):
    if not results.multi_hand_landmarks:
        return None

    hand = results.multi_hand_landmarks[0]
    puntos = []

    for lm in hand.landmark:
        puntos.extend([lm.x, lm.y, lm.z])

    if len(puntos) != FEATURES:
        return None

    return np.array(puntos, dtype=np.float32)


# ============================================================
# RECONOCEDOR ESTABLE
# ============================================================
cap = cv2.VideoCapture(0)

buffer_seq = []
frames_sin_mano = 0

pred_window = []          # Ventana de predicciones recientes
current_label = None      # Lo que mostramos en pantalla
ultimo_cambio = time.time()

print("\n🤟 Reconocedor dinámico SUPER ESTABLE activado.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # -------------------------------------------
    # Mano detectada
    # -------------------------------------------
    if results.multi_hand_landmarks:
        frames_sin_mano = 0

        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        puntos = extraer_landmarks(results)
        if puntos is not None:
            buffer_seq.append(puntos)
            if len(buffer_seq) > SEQ_LEN:
                buffer_seq.pop(0)

        # Secuencia lista
        if len(buffer_seq) == SEQ_LEN:

            seq = torch.tensor(
                np.array(buffer_seq, dtype=np.float32)
            ).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(seq)
                idx = pred.argmax(dim=1).item()
                label = id_to_label[idx]

            # Agregar pred a ventana
            pred_window.append(label)
            if len(pred_window) > WINDOW_STABLE:
                pred_window.pop(0)

            # Si todas las últimas N predicciones coinciden → actualizar
            if len(pred_window) == WINDOW_STABLE and all(p == pred_window[0] for p in pred_window):
                if pred_window[0] != current_label:
                    # Cambio REAL detectado
                    current_label = pred_window[0]
                    ultimo_cambio = time.time()
                    print(f"✔ Nueva seña: {current_label}")

    # -------------------------------------------
    # Mano perdida temporalmente
    # -------------------------------------------
    else:
        frames_sin_mano += 1

        if frames_sin_mano >= PERDIDA_TOLERANCIA:
            pred_window = []   # limpiar ruido
            # pero NO borrar el current_label → mantiene la seña

    # -------------------------------------------
    # Mostrar la seña ACTUAL estable
    # -------------------------------------------
    if current_label is not None:
        cv2.putText(frame, f"Senia: {current_label}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)

    cv2.imshow("Reconocedor Estable Dinamico - Presione ESC para Salir...", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
