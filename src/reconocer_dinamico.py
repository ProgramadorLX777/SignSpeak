import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import joblib
import time

# -------------------------
# CONFIG
# -------------------------
SEQ_LEN = 20
FEATURES = 84
MODEL_PATH = "models/modelo_dinamico.pth"
LABELS_PATH = "models/labels_dinamicos.pkl"

id_to_label = joblib.load(LABELS_PATH)
num_classes = len(id_to_label)

# -------------------------
# MODELO
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

modelo = ModeloLSTM(FEATURES, 128, num_classes)
modelo.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
modelo.eval()

# -------------------------
# MEDIAPIPE
# -------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

secuencia = []

print("🎥 Reconociendo señas dinámicas... ESC para salir")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    texto = "Esperando secuencia..."

    if results.multi_hand_landmarks:
        mano = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, mano, mp_hands.HAND_CONNECTIONS)

        wrist_x = mano.landmark[0].x
        wrist_y = mano.landmark[0].y

        datos = []
        for lm in mano.landmark:
            datos.extend([lm.x - wrist_x, lm.y - wrist_y])

        while len(datos) < FEATURES:
            datos.append(0.0)

        secuencia.append(datos)

        # Cuando la secuencia llega a 20 frames → predecimos
        if len(secuencia) == SEQ_LEN:
            seq_tensor = torch.tensor([secuencia], dtype=torch.float32)

            with torch.no_grad():
                out = modelo(seq_tensor)
                idx = torch.argmax(out).item()
                texto = id_to_label[idx]

            secuencia = []   # reiniciar para nueva secuencia

    cv2.putText(frame, texto, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imshow("Reconocimiento dinámico", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
