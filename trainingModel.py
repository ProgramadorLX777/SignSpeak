import os
import csv
import numpy as np
import tkinter as tk
from sklearn.neighbors import KNeighborsClassifier
from tkinter import messagebox
import joblib

X = []
y = []

for archivo in os.listdir("datos"):
    if archivo.endswith(".csv"):
        etiqueta = archivo.replace(".csv", "")
        with open(f"datos/{archivo}", "r") as f:
            reader = csv.reader(f)
            for fila in reader:
                X.append([float(x) for x in fila])
                y.append(etiqueta)

modelo = KNeighborsClassifier(n_neighbors=3)
modelo.fit(X, y)

joblib.dump(modelo, "modelo_senas.pkl")
root = tk.Tk()
root.withdraw()  # Oculta la ventana principal
messagebox.showinfo("Éxito", "Modelo entrenado y guardado correctamente!")