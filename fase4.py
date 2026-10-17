# -*- coding: utf-8 -*-
import sys
import subprocess
import os
import shutil

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPER_MODEL = os.path.join(BASE_DIR, "voces", "es_ES-davefx-medium.onnx")
PIPER_EXE = shutil.which("piper") or "piper"

def sintetizar(texto, salida="respuesta.wav"):
    if not os.path.exists(PIPER_MODEL):
        raise FileNotFoundError(f"Falta el modelo en {PIPER_MODEL}")
        
    subprocess.run(
        [PIPER_EXE, "--model", PIPER_MODEL, "--output_file", salida],
        input=texto.encode("utf-8"),
        check=True
    )
    return salida

if __name__ == "__main__":
    texto = "Voy a llevar la taza roja a la cocina."
    print(f"🔊 Sintetizando: {texto}")
    archivo = sintetizar(texto)
    print(f"✅ Guardado en: {archivo}")