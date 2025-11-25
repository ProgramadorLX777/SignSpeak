# src/utils.py
import os
import csv
import pickle

def ensure_dir(path):
    """Crea una carpeta si no existe."""
    if not os.path.exists(path):
        os.makedirs(path)

def save_csv(filepath, data_rows):
    """Guarda una lista de filas (list of lists) en CSV."""
    with open(filepath, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data_rows)

def load_model(model_path):
    """Carga un modelo entrenado .pkl"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def save_model(model, model_path):
    """Guarda un modelo entrenado .pkl"""
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
