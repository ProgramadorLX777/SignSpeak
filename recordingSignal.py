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
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

if not os.path.exists("datos"):
    os.makedirs("datos")
    
nombre_sena = simpledialog.askstring("Seña: ", "Ingrese el nombre de la Seña: ")

if nombre_sena:
    print(f"Se grabará la seña: {nombre_sena}")
else:
    messagebox.showerror("ERROR!!", "NO SE PUDO INGRESAR EL NOMBRE DE LA SEÑA!!")
    
archivo = f"datos/{nombre_sena}.csv"
num_frames = 100

cap = cv2.VideoCapture(0)
contador = 0
grabando = False
tiempo_mano_detectada = None  # Para contar los 5 segundos de detección estable
ultimo_visto = time.time()    # Última vez que se vio la mano
pausado = False                # indica si la grabación está pausada


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
            
            # Si aún no está grabando, muestra cuenta regresiva
            if not grabando:
                segundos_restantes = int(5 - tiempo_transcurrido)
                if segundos_restantes > 0:
                    cv2.putText(frame, f"Iniciando en: {segundos_restantes}s", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
                else:
                    grabando = True
                    contador = 0  # 🚀 Reiniciamos desde cero al empezar realmente
                    messagebox.showinfo("Grabando", "Comienza la grabación de la seña.")
            
             # --- SOLO GRABA SI grabando == True ---
            if grabando:
                datos = []
                for hand in results.multi_hand_landmarks:
                    wrist_x, wrist_y, wrist_z = hand.landmark[0].x, hand.landmark[0].y, hand.landmark[0].z
                    for lm in hand.landmark:
                        datos.extend([
                            (lm.x - wrist_x),
                            (lm.y - wrist_y),
                            (lm.z - wrist_z)
                        ])
                while len(datos) < 84:
                    datos.append(0.0)
                writer.writerow(datos)
                contador += 1

                for handLms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            ultimo_visto = time.time()  # actualiza el último momento en que se vio la mano

        else:
            # Si no hay manos visibles
            cv2.putText(frame, "No se detecta ninguna mano", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # --- Reinicia el contador si la mano desaparece más de 3s ---
            if grabando and (time.time() - ultimo_visto > 3):
                grabando = False
                tiempo_mano_detectada = None
                contador = 0  # 🚫 Reinicia los frames grabados
                cv2.putText(frame, "Grabación pausada por falta de mano", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                messagebox.showwarning("Pausa", "Se perdió la mano. Grabación pausada.")
                
            # Si la mano desaparece antes de iniciar la grabación (durante la cuenta regresiva)
            if not grabando and tiempo_mano_detectada is not None:
                tiempo_mano_detectada = None  # 🔄 Reinicia la cuenta regresiva    
                
        # Mostrar progreso
        cv2.putText(frame, f"{nombre_sena} ({contador}/100)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Grabando seña: ", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
messagebox.showinfo("Grabacion Seña", "Seña almacenada con éxito!!!")
