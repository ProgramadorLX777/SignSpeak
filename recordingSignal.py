import cv2
import mediapipe as mp
import numpy as np
import csv
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

if not os.path.exists("datos"):
    os.makedirs("datos")

nombre_sena = input("¿Qué seña quiere grabar?: ")
archivo = f"datos/{nombre_sena}.csv"
num_frames = 100

cap = cv2.VideoCapture(0)
contador = 0

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
            datos = []
            for hand in results.multi_hand_landmarks:
                for lm in hand.landmark:
                    datos.extend([lm.x, lm.y])
            while len(datos) < 84:
                datos.append(0.0)
            writer.writerow(datos)
            contador += 1

            cv2.putText(frame, f"{nombre_sena} ({contador}/100)", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Grabando seña: ", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Seña: '{nombre_sena}' guardada en datos/{nombre_sena}.csv")
