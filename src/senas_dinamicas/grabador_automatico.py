import cv2
import os
import time
import numpy as np
from mediapipe import solutions as mp

# ============================================
# CONFIGURACIÓN
# ============================================
DATA_DIR = "data_dynamic"
SEQ_LEN = 50
CAPTURAS_POR_CLASE = 5  # Número de secuencias para pruebas rápidas

mp_hands = mp.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ============================================
# CREAR DIRECTORIO
# ============================================
def asegurar_directorio(label):
    carpeta = os.path.join(DATA_DIR, label)
    os.makedirs(carpeta, exist_ok=True)
    return carpeta

# ============================================
# EXTRAER 63 FEATURES MANO
# ============================================
def extraer_landmarks(results):
    if not results.multi_hand_landmarks:
        return None

    coords = []
    for lm in results.multi_hand_landmarks[0].landmark:
        coords.extend([lm.x, lm.y, lm.z])

    return coords if len(coords) == 63 else None

# ============================================
# PROCESO DE GRABACIÓN
# ============================================
def grabar_senas(label):
    carpeta = asegurar_directorio(label)

    cap = cv2.VideoCapture(0)
    
    print("\n=================================================")
    print(f"   Preparando grabación para la seña: {label}")
    print("=================================================\n")

    time.sleep(1)

    for i in range(CAPTURAS_POR_CLASE):
        print(f"\n📌 Secuencia {i+1}/{CAPTURAS_POR_CLASE}")
        
        # Cuenta regresiva
        for t in [3, 2, 1]:
            print(f"Comenzando en {t}...")
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)  # <-- Corregir espejo
            cv2.putText(frame, f"{t}", (250, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
            cv2.imshow("Grabando", frame)
            cv2.waitKey(1000)

        print("🎥 GRABANDO...")
        frames = []

        while len(frames) < SEQ_LEN:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)  # <-- Corregir espejo
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            landmarks = extraer_landmarks(results)
            if landmarks is not None:
                frames.append(landmarks)

            # UI
            cv2.putText(frame, f"Grabando {label}: {len(frames)}/{SEQ_LEN}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Grabando", frame)
            if cv2.waitKey(1) == 27:
                break

        # Guardar secuencia
        file_name = os.path.join(carpeta, f"{int(time.time()*1000)}.npy")
        np.save(file_name, np.array(frames))
        print(f"✔ Secuencia guardada: {file_name}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n🎉 Completado!")
    print(f"Se grabaron {CAPTURAS_POR_CLASE} secuencias para '{label}'")

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("=== Sistema automático de grabación de señas ===\n")
    label = input("Ingresa el nombre de la seña a grabar: ").strip()
    grabar_senas(label)
