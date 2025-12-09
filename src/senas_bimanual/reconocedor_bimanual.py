import cv2
import mediapipe as mp
import numpy as np
import torch
import joblib
import time
from collections import deque

MODEL_PATH = "models/modelo_dinamico_bimanual.pth"
LABELS_PATH = "models/labels_bimano.pkl"

SEQ_LEN = 50
FEATURES = 126

# umbrales y tiempos (no los toqué excepto si quieres ajustar)
UMBRAL_CONF = 0.70
VENTANA_ESTABILIDAD = 5
TIEMPO_DESAPARECER = 2.0

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
id_to_label = joblib.load(LABELS_PATH)

ultimo_timestamp = time.time()
ultimo_resultado = ""
historial = []

# -----------------------------
# MODELO TRANSFORMER (igual al tuyo)
# -----------------------------
class TransformerBimano(torch.nn.Module):
    def __init__(self, input_dim, num_classes, dim=256, heads=8, layers=4):
        super().__init__()

        self.input_fc = torch.nn.Linear(input_dim, dim)

        encoder = torch.nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = torch.nn.TransformerEncoder(encoder, num_layers=layers)
        self.fc = torch.nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.input_fc(x)
        out = self.transformer(x)
        last = out[:, -1, :]
        return self.fc(last)

model = TransformerBimano(FEATURES, len(id_to_label))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# -----------------------------
# DIBUJO LANDMARKS (colores personalizados)
# -----------------------------
def dibujar_manos_colores(frame, hand_landmarks):
    # conexiones en cian
    for connection in mp_hands.HAND_CONNECTIONS:
        x1 = int(hand_landmarks.landmark[connection[0]].x * frame.shape[1])
        y1 = int(hand_landmarks.landmark[connection[0]].y * frame.shape[0])
        x2 = int(hand_landmarks.landmark[connection[1]].x * frame.shape[1])
        y2 = int(hand_landmarks.landmark[connection[1]].y * frame.shape[0])
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # puntos con colores según índice / zona
    for idx, lm in enumerate(hand_landmarks.landmark):
        x = int(lm.x * frame.shape[1])
        y = int(lm.y * frame.shape[0])

        if idx in [4, 8, 12, 16, 20]:       # puntas dedos
            color = (0, 255, 0)
        elif idx in [3, 7, 11, 15, 19]:     # articulaciones medias
            color = (255, 0, 0)
        elif idx == 0:                      # muñeca / base
            color = (0, 0, 255)
        else:                               # demás
            color = (0, 165, 255)

        cv2.circle(frame, (x, y), 5, color, -1)

# -----------------------------
# RECONOCIMIENTO (con NO DETECTADO + ordenamiento left/right)
# -----------------------------
window = deque(maxlen=SEQ_LEN)
cap = cv2.VideoCapture(0)

print("Reconociendo señas bimanuales (Transformer)…")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    flip = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(flip, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Default zero hands (left, right)
    mano_left = np.zeros((21, 3))
    mano_right = np.zeros((21, 3))

    # Dibujar y asignar manos por handedness si está disponible
    manos_detectadas = False

    if result.multi_hand_landmarks:
        # dibuja todos (si hay 1 o 2) usando la función de colores
        for hl in result.multi_hand_landmarks:
            dibujar_manos_colores(flip, hl)

    # ordenamiento por handedness (cuando está disponible)
    if getattr(result, "multi_hand_landmarks", None) and getattr(result, "multi_handedness", None):
        # zip está seguro porque multi_hand_landmarks y multi_handedness se alinean en mediapipe
        for lm, hand_h in zip(result.multi_hand_landmarks, result.multi_handedness):
            pts = np.array([[p.x, p.y, p.z] for p in lm.landmark])
            label = hand_h.classification[0].label  # 'Left' o 'Right'
            if label == "Left":
                mano_left = pts
            else:
                mano_right = pts

        # consideramos detectado sólo si tenemos ambas manos (para este caso bimanual)
        if (not np.allclose(mano_left, 0)) and (not np.allclose(mano_right, 0)):
            manos_detectadas = True

    # Si no hay multi_handedness (p. ej. sólo una mano), intentamos asignar por X (fallback)
    elif result.multi_hand_landmarks:
        # clasificamos por coordenada x: menor x -> left (en imagen no flip), pero como hacemos flip, invertimos
        # Aquí usamos el x después de flip = reflejado, así que menor x corresponde a mano izquierda del usuario
        landmarks_list = []
        for lm in result.multi_hand_landmarks:
            pts = np.array([[p.x, p.y, p.z] for p in lm.landmark])
            landmarks_list.append(pts)

        if len(landmarks_list) == 2:
            # ordenar por coordenada x media
            xs = [pts[:,0].mean() for pts in landmarks_list]
            if xs[0] <= xs[1]:
                mano_left = landmarks_list[0]
                mano_right = landmarks_list[1]
            else:
                mano_left = landmarks_list[1]
                mano_right = landmarks_list[0]
            manos_detectadas = True
        else:
            # solo 1 mano presente -> no bimanual detectado
            manos_detectadas = False

    # vector (left then right) — coincide con tu entrenamiento
    vec = np.concatenate([mano_left.flatten(), mano_right.flatten()])
    window.append(vec)

    ahora = time.time()

    # Si no detectamos ambas manos, mostramos NO DETECTADO y reseteamos historial
    if not manos_detectadas:
        ultimo_resultado = "NO DETECTADO"
        historial = []
        ultimo_timestamp = ahora

    # Si la ventana está llena y detectamos manos bimanuales, predecimos
    if len(window) == SEQ_LEN and manos_detectadas:
        seq = np.array(window, dtype=np.float32)
        seq = (seq - np.mean(seq)) / (np.std(seq) + 1e-6)
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(seq_tensor)
            probs = torch.softmax(logits, dim=1)[0]

        pred_id = torch.argmax(probs).item()
        pred_label = id_to_label[pred_id]
        conf = probs[pred_id].item()

        # estabilidad: solo aceptar si supera umbral y se repite en la ventana de historial
        if conf >= UMBRAL_CONF:
            historial.append(pred_label)
            if len(historial) >= VENTANA_ESTABILIDAD:
                if historial.count(pred_label) >= VENTANA_ESTABILIDAD:
                    ultimo_resultado = pred_label
                    ultimo_timestamp = ahora
                    historial = []
        else:
            # confianza baja -> no aceptamos, limpiamos historial
            historial = []

    # limpiar texto si no hay detección por más del tiempo configurado
    if ahora - ultimo_timestamp > TIEMPO_DESAPARECER:
        ultimo_resultado = ""

    # mostrar resultado (puedes ajustar la posición/estilo)
    cv2.putText(flip, ultimo_resultado, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

    cv2.imshow("Reconocedor bimanual (Transformer) - Presione ESC para Salir...", flip)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()