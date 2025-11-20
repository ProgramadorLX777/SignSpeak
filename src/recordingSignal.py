import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time
import tkinter as tk
from tkinter import simpledialog, messagebox

root = tk.Tk()
root.withdraw()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, 
                       max_num_hands=1,       # ⬅ SOLO UNA MANO (PyTorch)
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

if not os.path.exists("datos"):
    os.makedirs("datos")
    
nombre_sena = simpledialog.askstring("Seña para grabar: ", "Ingrese el nombre de la Seña: ")

if nombre_sena:
    print(f"Se grabará la seña: {nombre_sena}")
else:
    messagebox.showerror("ERROR!!", "NO SE PUDO INGRESAR EL NOMBRE DE LA SEÑA!!")
    exit()
        
archivo = f"datos/{nombre_sena}.csv"
num_frames = 300

cap = cv2.VideoCapture(0)
contador = 0
grabando = False
tiempo_mano_detectada = None   # Para contar los 5 segundos de detección estable
ultimo_visto = time.time()     # Última vez que se vio la mano
pausado = False                # indica si la grabación está pausada
prev_filtered = None  # almacena landmarks suavizados
alpha = 0.2           # factor de suavizado EMA (ajustable)
cancelado = False
mano_detectada_alguna_vez = False

def suavizar_landmarks(new_points, prev_points, alpha=0.2):
    """Suaviza los landmarks usando Exponential Moving Average (EMA)."""
    if prev_points is None:
        return new_points
    return (alpha * new_points) + ((1 - alpha) * prev_points)

print(f"📸 Grabando la seña: '{nombre_sena}'...")

with open(archivo, mode='w', newline='') as f:
    writer = csv.writer(f)
    
    while contador < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:

            # Si es la primera vez que detecta la mano, marca el tiempo
            if tiempo_mano_detectada is None:
                tiempo_mano_detectada = time.time()
            
            # Calcula cuánto tiempo lleva viendo la mano
            tiempo_transcurrido = time.time() - tiempo_mano_detectada
            
            # Cuenta regresiva ANTES de grabar
            if not grabando:
                segundos_restantes = int(5 - tiempo_transcurrido)
                
                if segundos_restantes > 0:
                    cv2.putText(frame, f"Iniciando en: {segundos_restantes}s", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 6)

                    cv2.putText(frame, f"Iniciando en: {segundos_restantes}s", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,0), 3)
                else:
                    grabando = True
                    contador = 0
                    print("Grabando Seña!!")

            # --- SOLO GRABA SI grabando == True ---
            if grabando:
                mano = results.multi_hand_landmarks[0]

                # 1) Extraemos los 63 valores crudos
                raw = np.array([[lm.x, lm.y, lm.z] for lm in mano.landmark]).flatten()

                # 2) Aplicamos suavizado temporal EMA
                filtered = suavizar_landmarks(raw, prev_filtered, alpha)
                prev_filtered = filtered.copy()

                datos = []

                # 3) Normalizamos respecto a la muñeca usando valores SUAVIZADOS
                wrist_x = filtered[0]
                wrist_y = filtered[1]
                wrist_z = filtered[2]

                for i in range(0, len(filtered), 3):
                    nx = filtered[i]   - wrist_x
                    ny = filtered[i+1] - wrist_y
                    nz = filtered[i+2] - wrist_z
                    datos.extend([nx, ny, nz])

                # 4) Mantener EXACTAMENTE tu forma fija (126 valores)
                while len(datos) < 126:
                    datos.append(0.0)

                writer.writerow(datos)
                contador += 1

                mp_draw.draw_landmarks(frame, mano, mp_hands.HAND_CONNECTIONS)

            ultimo_visto = time.time()
            
        else:
            # Si no hay manos visibles
            cv2.putText(frame, "No se detecta ninguna mano", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 6)

            cv2.putText(frame, "No se detecta ninguna mano", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            # --- Reinicia si la mano desaparece más de 3 segundos ---
            if grabando and (time.time() - ultimo_visto > 3):
                grabando = False
                tiempo_mano_detectada = None
                contador = 0  # Reinicia frames
                cv2.putText(frame, "Grabación pausada por falta de mano", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                messagebox.showwarning("Pausa", "Se perdió la mano. Grabación pausada.")

            # Si desaparece antes de grabar (durante la cuenta)
            if not grabando and tiempo_mano_detectada is not None:
                tiempo_mano_detectada = None

        # Mostrar progreso
        cv2.putText(frame, f"{nombre_sena} ({contador}/300)", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)
        cv2.putText(frame, f"{nombre_sena} ({contador}/300)", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        cv2.imshow("Grabando frames - Presione ESC para Salir: ", frame)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 27:  # ESC
            cancelado = True
            break
        # FALTA AGREGAR VALIDACION PARA QUE PESE A QUE NO GUARDE NADA NO MUESTRE EL MENSAJE

cap.release()
cv2.destroyAllWindows()
messagebox.showinfo("Grabacion Seña", "Seña almacenada con éxito!!!")
