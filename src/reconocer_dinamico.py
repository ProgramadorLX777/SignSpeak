import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import joblib
import time

SEQ_LEN = 20
FEATURES = 63

MODEL_PATH = "models/modelo_dinamico.pth"
LABELS_PATH = "models/labels_dinamicos.pkl"

ultimo_texto = "Esperando secuencia..."

id_to_label = joblib.load(LABELS_PATH)
num_classes = len(id_to_label)

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

    texto = ultimo_texto

    if results.multi_hand_landmarks:
        mano = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, mano, mp_hands.HAND_CONNECTIONS)

        datos = []
        for lm in mano.landmark:
            datos.extend([lm.x, lm.y, lm.z])  # <--- las 63 features correctas

        secuencia.append(datos)

        if len(secuencia) == SEQ_LEN:
            seq_tensor = torch.tensor([secuencia], dtype=torch.float32)

            with torch.no_grad():
                out = modelo(seq_tensor)
                prob = torch.softmax(out, dim=1)
                conf, idx = torch.max(prob, dim=1)
                conf = conf.item()
                idx = idx.item()

            if conf >= 0.60:  # puedes ajustar el threshold
                ultimo_texto = f"{id_to_label[idx]} ({conf:.2f})"
            else:
                ultimo_texto = "Confianza baja..."

            secuencia = []  # reiniciar


    cv2.putText(frame, texto, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imshow("Reconocimiento dinámico", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
