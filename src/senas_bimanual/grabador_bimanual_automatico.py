import cv2
import mediapipe as mp
import numpy as np
import os
import time

DATA_DIR = "data_dynamic_bimano"
SEQ_LEN = 50
CAPTURAS = 5
FEATURES = 126

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

os.makedirs(DATA_DIR, exist_ok=True)

def asegurar_dir(label):
    carpeta = os.path.join(DATA_DIR, label)
    os.makedirs(carpeta, exist_ok=True)
    return carpeta

def extraer_landmarks(result):
    mano1 = np.zeros((21,3))
    mano2 = np.zeros((21,3))

    if result.multi_hand_landmarks:
        for i, lm in enumerate(result.multi_hand_landmarks):
            pts = []
            for p in lm.landmark:
                pts.append([p.x, p.y, p.z])

            if i == 0:
                mano1 = np.array(pts)
            elif i == 1:
                mano2 = np.array(pts)

    return np.concatenate([mano1.flatten(), mano2.flatten()]) 

def grabar(label):
    carpeta = asegurar_dir(label)
    cap = cv2.VideoCapture(0)

    print(f"\n=== Grabando seña '{label}' ===")

    for i in range(CAPTURAS):
        print(f"\nSecuencia {i+1}/{CAPTURAS}")

        for t in [3,2,1]:
            print(f"Inicia en {t}…")
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, str(t), (250,250),
                        cv2.FONT_HERSHEY_SIMPLEX, 3,(0,255,0),5)
            cv2.imshow("Grabando", frame)
            cv2.waitKey(1000)

        print("Grabando frames…")
        secuencia = []

        while len(secuencia) < SEQ_LEN:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = hands.process(rgb)
            landmarks = extraer_landmarks(result)

            secuencia.append(landmarks)

            cv2.putText(frame, f"{len(secuencia)}/{SEQ_LEN}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
            cv2.imshow("Grabando", frame)

            if cv2.waitKey(1) == 27:
                break

        secuencia = np.array(secuencia)
        secuencia = (secuencia - np.mean(secuencia)) / (np.std(secuencia) + 1e-6)

        nombre = os.path.join(carpeta, f"{int(time.time()*1000)}.npy")
        np.save(nombre, secuencia)

        print("Guardado:", nombre)

    cap.release()
    cv2.destroyAllWindows()
    print("\n✔ Grabación completa.")

if __name__ == "__main__":
    label = input("Nombre de la seña: ").strip()
    grabar(label)
