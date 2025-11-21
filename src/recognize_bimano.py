import cv2
import mediapipe as mp
import numpy as np
import torch
import joblib
from collections import deque
import time

MODEL_PATH = "models/modelo_bimano.pth"
LABELS_PATH = "models/labels_bimano.pkl"

SEQ_LEN = 30
FEATURES = 126

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar labels
id_to_label = joblib.load(LABELS_PATH)

# Modelo
class LSTMBimano(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

model = LSTMBimano(FEATURES, 128, len(id_to_label))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Buffer sliding window
window = deque(maxlen=SEQ_LEN)

cap = cv2.VideoCapture(0)

print("Reconociendo señas bimano…")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    flip = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(flip, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    mano1 = np.zeros((21, 3))
    mano2 = np.zeros((21, 3))

    if result.multi_hand_landmarks:
        
        # === DIBUJAR AMBAS MANOS ===
        for hl in result.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                flip, 
                hl, 
                mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style()
            )

        # === EXTRAER LANDMARKS PARA EL MODELO ===
        for i, lm in enumerate(result.multi_hand_landmarks):
            pts = []
            for p in lm.landmark:
                pts.append([p.x, p.y, p.z])

            if i == 0:
                mano1 = np.array(pts)
            elif i == 1:
                mano2 = np.array(pts)

    # === UNIR VECTORES ===
    feature_vector = np.concatenate([mano1.flatten(), mano2.flatten()])
    window.append(feature_vector)

    # === PREDICCIÓN ===
    if len(window) == SEQ_LEN:
        seq = np.array(window, dtype=np.float32)
        seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(seq)
            probs = torch.softmax(logits, dim=1)[0]
            pred_id = torch.argmax(probs).item()
            pred_label = id_to_label[pred_id]
            conf = probs[pred_id].item()

        cv2.putText(flip, f"{pred_label} ({conf:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Reconocedor", flip)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
