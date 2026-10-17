# -*- coding: utf-8 -*-
import sys, os, json, time, queue, subprocess, shutil
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from faster_whisper import WhisperModel
from fase2_json import extraer_json

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPER_MODEL = os.path.join(BASE_DIR, "voces", "es_ES-davefx-medium.onnx")
PIPER_EXE = shutil.which("piper") or "piper"

print("⏳ Cargando Whisper...")
modelo_whisper = WhisperModel("small", device="cpu", compute_type="int8")

cola = queue.Queue()
def callback(indata, frames, time_info, status):
    cola.put(indata.copy())

def rms(b): return float(np.sqrt(np.mean(b ** 2)))

def a_whisper(audio):
    audio = audio.flatten().astype(np.float32)
    if 48000 != 16000:
        audio = resample_poly(audio, 16000//np.gcd(48000,16000), 48000//np.gcd(48000,16000)).astype(np.float32)
    return audio / (np.max(np.abs(audio)) + 1e-8)

def generar_respuesta(json_str):
    data = json.loads(json_str)
    acciones = data.get("acciones", [])
    if not acciones: return "No entendí ninguna orden."
    partes = []
    for acc in acciones:
        obj = acc["objetos"][0]["nombre"] if acc.get("objetos") else ""
        lugar = acc["lugares"][0]["nombre"] if acc.get("lugares") else ""
        persona = acc["personas"][0]["nombre"] if acc.get("personas") else ""
        partes.append(f"Voy a {acc['accion']} {obj} {lugar} {persona}".strip())
    return " y ".join(partes)

def main():
    print("🎤 Habla en español. Ctrl+C para salir.\n")
    buffer, hablando, silencio = [], False, 0
    
    with sd.InputStream(samplerate=48000, blocksize=int(48000*0.03), channels=1, dtype="float32", callback=callback):
        while True:
            bloque = cola.get()
            if rms(bloque) > 0.01:
                if not hablando: hablando = True; print("🔴 ", end="", flush=True)
                buffer.append(bloque); silencio = 0
            else:
                if hablando:
                    buffer.append(bloque); silencio += 1
                    if silencio >= int(0.9/0.03):
                        if len(buffer) >= int(0.4/0.03):
                            audio = np.concatenate(buffer, axis=0)
                            texto = " ".join(s.text for s in modelo_whisper.transcribe(a_whisper(audio), language="es")[0])
                            print(f"\r📝 Tú: {texto}")
                            if texto:
                                json_str = extraer_json(texto)
                                print(f"📦 JSON: {json_str}")
                                respuesta = generar_respuesta(json_str)
                                print(f"🤖 Robot: {respuesta}")
                                subprocess.run([PIPER_EXE, "--model", PIPER_MODEL, "--output_file", "respuesta.wav"], input=respuesta.encode(), check=True)
                                os.startfile("respuesta.wav")
                        buffer, hablando, silencio = [], False, 0

if __name__ == "__main__":
    main()