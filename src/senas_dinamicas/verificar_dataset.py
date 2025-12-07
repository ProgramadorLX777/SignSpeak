import os
import numpy as np

DATA_DIR = "data_dynamic"

for label in os.listdir(DATA_DIR):
    carpeta = os.path.join(DATA_DIR, label)
    if not os.path.isdir(carpeta):
        continue

    archivos = [f for f in os.listdir(carpeta) if f.endswith(".npy")]

    print(f"\n=== {label.upper()} ===")
    print("Total secuencias:", len(archivos))

    for f in archivos[:3]:  # revisa solo los primeros 3
        arr = np.load(os.path.join(carpeta, f))
        print(f"{f}: shape {arr.shape}")
