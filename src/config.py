# src/config.py
import os

# --- Rutas base ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# --- Carpetas ---
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
MODEL_PATH = os.path.join(BASE_DIR, "models", "modelo_senas.pkl")

# --- Parámetros globales ---
NUM_FRAMES = 100          # cantidad de frames por seña
STABLE_TIME = 5           # segundos que debe mantenerse la mano antes de grabar
MISSING_TIME = 3          # segundos que puede perder la mano antes de pausar grabación
CONF_THRESHOLD = 0.7      # confianza mínima de detección
