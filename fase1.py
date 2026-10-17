# -*- coding: utf-8 -*-
"""
Fase 1: Voz a Texto (S2T) con faster-whisper.
"""
import sys
import io
import os
from faster_whisper import WhisperModel

# Configurar consola para UTF-8 en Windows
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Configuración
WHISPER_MODEL = "small"
AUDIO_PATH = "audios/orden_001.wav"

print(" Cargando Whisper...")
# device="cpu" y compute_type="int8" son clave para que corra fluido sin GPU
modelo = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
print(" Modelo cargado.\n")

def transcribir(ruta_audio):
    segments, info = modelo.transcribe(
        ruta_audio,
        language="es",
        beam_size=5,
        temperature=0.0,
        vad_filter=True  # Filtra silencios
    )
    texto = " ".join([seg.text for seg in segments])
    return texto.strip()

if __name__ == "__main__":
    if not os.path.exists(AUDIO_PATH):
        print(f"❌ No se encuentra el archivo: {AUDIO_PATH}")
    else:
        print(f"🎧 Procesando: {AUDIO_PATH}")
        resultado = transcribir(AUDIO_PATH)
        print(f"📝 Transcripción: {resultado}")