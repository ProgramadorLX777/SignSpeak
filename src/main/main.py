import tkinter as tk
from tkinter import messagebox
import subprocess

def grabar():
    subprocess.Popen(["python", "src/recordingSignal.py"])

def entrenar():
    proceso = subprocess.run(["python", "src/training_model_pytorch.py"], capture_output=True, text=True)
    # Revisar esta linea para mensaje
    messagebox.showinfo("Entrenamiento terminado", proceso.stdout)
    
def reconocer():
    subprocess.Popen(["python", "src/recognizeSignal.py"])
    
def grabar_movimiento():
    subprocess.Popen(["python", "src/grabacion_con_movimiento.py"])

root = tk.Tk()
root.title("Sistema de Senas")
root.geometry("300x200")

btn_grabar = tk.Button(root, text="Grabar Seña", command=grabar, width=20, height=2)
btn_grabar.pack(pady=10)

btn_entrenar = tk.Button(root, text="Entrenar Modelo", command=entrenar, width=20, height=2)
btn_entrenar.pack(pady=10)

btn_reconocer = tk.Button(root, text="Reconocer Señas", command=reconocer, width=20, height=2)
btn_reconocer.pack(pady=10)

btn_grabar_movimiento = tk.Button(root, text="Grabar Movimientos", command=grabar_movimiento, width=20, height=2)
btn_grabar_movimiento.pack(pady=10)

root.mainloop()
