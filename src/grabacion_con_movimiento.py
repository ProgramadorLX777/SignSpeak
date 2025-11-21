import cv2
import mediapipe as mp
import numpy as np
import os
import time
import tkinter as tk
from tkinter import simpledialog, messagebox

# ------------------ TKINTER CONFIG ------------------
root = tk.Tk()
root.withdraw()

sign_name = simpledialog.askstring("Seña dinámica", "Ingrese el nombre de la seña a grabar:")

if not sign_name:
    messagebox.showerror("Error", "Debe ingresar un nombre para la seña.")
    raise SystemExit

# -----------------------------------------------------

mp_hands = mp.solutions.hands

# ---------- NORMALIZAR SECUENCIA A LONGITUD FIJA ----------
FIXED_LENGTH = 50

def normalize_sequence(seq, target_len=FIXED_LENGTH):
    seq = np.array(seq)

    if len(seq) > target_len:
        # Si hay demasiados → recortar equiespaciado
        indices = np.linspace(0, len(seq) - 1, target_len).astype(int)
        seq = seq[indices]
    else:
        # Si faltan → repetir últimos frames
        while len(seq) < target_len:
            seq = np.vstack([seq, seq[-1]])

    return seq

def record_dynamic_sign(sign_name, output_dir="data_dynamic", record_seconds=5):
    """
    Graba la seña dinámica como una única secuencia .npy
    compatible con el modelo LSTM/BiLSTM.
    """

    # Crear carpetas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sign_path = os.path.join(output_dir, sign_name)
    if not os.path.exists(sign_path):
        os.makedirs(sign_path)

    cap = cv2.VideoCapture(0)

    print(f"\n▶ Aproxime su mano para comenzar la grabación de: '{sign_name}'")
    print("Esperando detección de mano...")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        # ---------- ESPERAR MANO ----------
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:

                # ---- CUENTA REGRESIVA ----
                for i in range(3, 0, -1):
                    ret2, frame2 = cap.read()
                    frame2 = cv2.flip(frame2, 1)

                    cv2.putText(frame2, f"Grabando en: {i}s",
                                (50, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                1.5, (0, 255, 0), 3)

                    cv2.imshow("Grabación de Seña Dinámica", frame2)
                    cv2.waitKey(1000)

                break

            cv2.putText(frame, "Acerque su mano para comenzar...",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 3)

            cv2.imshow("Grabación de Seña Dinámica", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                cap.release()
                cv2.destroyAllWindows()
                return

        # ---------- GRABACIÓN ----------
        print("\n🎥 Grabando movimiento...\n")
        start_time = time.time()

        sequence = []   # <--- secuencia completa para el modelo

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]

                landmarks = []
                for lm in hand.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                sequence.append(landmarks)

                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS
                )

            cv2.putText(frame, "Grabando movimiento...",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 255), 3)

            cv2.imshow("Grabación de Seña Dinámica", frame)

            if time.time() - start_time >= record_seconds:
                break

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

        # ---------- GUARDAR SECUENCIA COMPLETA ----------
        sequence = normalize_sequence(sequence)

        np.save(os.path.join(sign_path, f"{int(time.time())}.npy"), sequence)

        print(f"✔ Seña dinámica '{sign_name}' grabada.")
        print(f"✔ Frames en secuencia: {sequence.shape[0]}")

        messagebox.showinfo("Grabación completa",
                            f"Seña '{sign_name}' grabada.\n"
                            f"Frames capturados: {sequence.shape[0]}")


record_dynamic_sign(sign_name)
