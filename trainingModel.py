import os
import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
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
print("✅ Modelo entrenado guardado como 'modelo_senas.pkl'")
