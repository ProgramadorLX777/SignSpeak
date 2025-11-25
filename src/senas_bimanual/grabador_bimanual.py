import cv2
import mediapipe as mp
import numpy as np
import os
import time
import tkinter as tk
from tkinter import simpledialog, messagebox

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "data_dynamic_bimano"
SEQ_LEN = 50
FEATURES = 126  # 2 manos * 21 puntos * 3 coords

os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------
# OBTENER NOMBRE DE LA SEÑA
# -----------------------------
root = tk.Tk()
root.withdraw()

label = simpledialog.askstring("Nueva seña dinámica (2 manos)", "Nombre de la seña:")

if not label:
    messagebox.showerror("Error", "Debes ingresar un nombre.")
    exit()

carpeta = os.path.join(DATA_DIR, label)
os.makedirs(carpeta, exist_ok=True)

# Nombre secuencial
contador = len(os.listdir(carpeta)) + 1
ruta_salida = os.path.join(carpeta, f"{contador}.npy")

# -----------------------------
# INICIALIZAR MEDIAPIPE Y CAMARA
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    messagebox.showerror("Error", "No se pudo acceder a la cámara.")
    exit()

print("\n▶ Aproxime sus manos para comenzar la grabación…")
time.sleep(0.5)

# DESCARTAR FRAMES INICIALES
for _ in range(5):
    cap.read()

# -----------------------------
# ESPERAR DETECCIÓN DE MANO
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        # -----------------------
        # CUENTA REGRESIVA
        # -----------------------
        for i in range(3, 0, -1):
            ret2, frame2 = cap.read()
            frame2 = cv2.flip(frame2, 1)

            cv2.putText(frame2, f"Grabando en: {i}",
                        (50, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0, 255, 0), 4)

            cv2.imshow("Grabación 2 manos", frame2)
            cv2.waitKey(1000)

        break

    cv2.putText(frame, "Acerque sus manos...",
                (50, 80), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (0, 0, 255), 3)

    cv2.imshow("Grabación 2 manos", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        cap.release()
        cv2.destroyAllWindows()
        exit()

# -----------------------------
# GRABACIÓN EXACTA DE 30 FRAMES
# -----------------------------
print("\n🎥 Grabando secuencia…\n")
frames = []

while len(frames) < SEQ_LEN:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # manos por defecto (si falta una)
    mano1 = np.zeros((21, 3))
    mano2 = np.zeros((21, 3))

    if result.multi_hand_landmarks:
        for i, lm in enumerate(result.multi_hand_landmarks):
            pts = []
            for p in lm.landmark:
                pts.append([p.x, p.y, p.z])

            if i == 0:
                mano1 = np.array(pts)
            elif i == 1:
                mano2 = np.array(pts)

    # vector de 126 features
    vec = np.concatenate([mano1.flatten(), mano2.flatten()])
    frames.append(vec)

    cv2.putText(frame, f"Frames: {len(frames)}/{SEQ_LEN}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 255, 255), 3)

    cv2.imshow("Grabación 2 manos", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

frames = np.array(frames)

if frames.shape != (SEQ_LEN, FEATURES):
    messagebox.showerror("Error",
        f"Dimensiones incorrectas: {frames.shape}. No guardado.")
    exit()

np.save(ruta_salida, frames)

messagebox.showinfo(
    "Grabación completada",
    f"Seña '{label}' guardada.\nFrames: {SEQ_LEN}\nRuta:\n{ruta_salida}"
)

print(f"✔ Secuencia guardada en: {ruta_salida}")
