# grabador_dinamico_fix.py
import cv2, mediapipe as mp, numpy as np, os, time
import tkinter as tk
from tkinter import simpledialog, messagebox

root = tk.Tk(); root.withdraw()
sign_name = simpledialog.askstring("Seña dinámica", "Ingrese el nombre de la seña dinámica:")
if not sign_name:
    messagebox.showerror("Error", "Debe ingresar un nombre para la seña."); raise SystemExit

mp_hands = mp.solutions.hands

# ---------- PARÁMETROS COMUNES ----------
SEQ_LEN = 50            # debe coincidir con entrenador y reconocedor
FEATURES = 63
RECORD_SECONDS = 5
EMA_ALPHA = 0.25        # suavizado (0 = sin suavizado)
OUT_DIR = "data_dynamic"
# ---------------------------------------

def normalize_frame(landmarks):
    """Recibe lista de 21 landmarks (cada uno con x,y,z). Devuelve vector (63,) normalizado."""
    pts = np.array([[lm[0], lm[1], lm[2]] for lm in landmarks])  # asegurar shape (21,3)
    wrist = pts[0].copy()
    pts = pts - wrist  # centrar
    # escala referencia: distancia muñeca -> dedo medio (landmark 9 o 12)
    ref = np.linalg.norm(pts[9])  # 9 es middle_finger_mcp (puedes probar 12 para tip)
    if ref < 1e-6:
        ref = 1.0
    pts = pts / ref
    return pts.flatten()

def resample_sequence(seq, target_len=SEQ_LEN):
    seq = np.array(seq)
    if len(seq) == 0:
        return np.zeros((target_len, FEATURES))
    if len(seq) >= target_len:
        idx = np.linspace(0, len(seq)-1, target_len).astype(int)
        return seq[idx]
    # si faltan frames: repetir último
    last = seq[-1]
    pad = np.repeat(last[np.newaxis,:], target_len - len(seq), axis=0)
    return np.vstack([seq, pad])

def record_dynamic_sign(sign_name, output_dir=OUT_DIR, record_seconds=RECORD_SECONDS):
    os.makedirs(output_dir, exist_ok=True)
    sign_path = os.path.join(output_dir, sign_name); os.makedirs(sign_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    prev_filtered = None

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                        min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:

        # esperar mano
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame,1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            if res.multi_hand_landmarks:
                # cuenta regresiva 5s para usar igual que tu flujo
                for i in range(5,0,-1):
                    ret2, f2 = cap.read()
                    f2 = cv2.flip(f2,1)
                    cv2.putText(f2, f"Grabando en: {i}s",(40,60),cv2.FONT_HERSHEY_SIMPLEX,1.4,(0,255,0),3)
                    cv2.imshow("Grabando", f2); cv2.waitKey(1000)
                break
            cv2.putText(frame, "Acerque su mano para comenzar...", (20,60), cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)
            cv2.imshow("Grabando", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                cap.release(); cv2.destroyAllWindows(); return

        # grabar durante X segundos
        start = time.time()
        seq = []
        while time.time() - start < record_seconds:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame,1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                lm_list = [(lm.x, lm.y, lm.z) for lm in hand.landmark]
                vec = normalize_frame(lm_list)        # 63 vector normalized
                if prev_filtered is None or EMA_ALPHA <= 0:
                    filtered = vec
                else:
                    filtered = EMA_ALPHA * vec + (1-EMA_ALPHA) * prev_filtered
                prev_filtered = filtered
                seq.append(filtered)
                mp.solutions.drawing_utils.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            cv2.putText(frame, "Grabando movimiento...", (20,40), cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,255),2)
            cv2.imshow("Grabando", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release(); cv2.destroyAllWindows()

        seq = resample_sequence(seq, SEQ_LEN)   # asegurar len=SEQ_LEN
        ts = int(time.time())
        np.save(os.path.join(sign_path, f"{ts}.npy"), seq)
        messagebox.showinfo("Grabación completa", f"Seña '{sign_name}' grabada. Frames: {seq.shape[0]}")

if __name__ == "__main__":
    record_dynamic_sign(sign_name)
