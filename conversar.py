# -*- coding: utf-8 -*-
"""
conversar.py — Habla por el micrófono y el robot responde con lo que entendió.

Flujo:
    Micrófono -> Whisper -> Texto -> Ollama -> JSON -> Respuesta -> Piper -> Audio

Uso:
    python conversar.py

Requiere en la misma carpeta:
    - fase2_json.py
    - voces/es_ES-davefx-medium.onnx (y .onnx.json)
"""

import sys
import io
import os
import json
import time
import queue
import subprocess
import shutil

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from faster_whisper import WhisperModel

# ---------- UTF-8 en consola Windows ----------
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ---------- Importar el extractor de Fase 2 ----------
from fase2_json import extraer_json


# ============================================================
# 1. Configuración
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

WHISPER_MODEL = "small"
LLM_MODEL = "qwen2.5:1.5b"
PIPER_MODEL = os.path.join(BASE_DIR, "voces", "es_ES-davefx-medium.onnx")
PIPER_EXE = shutil.which("piper") or "piper"

SAMPLE_RATE_MIC = 48000
SAMPLE_RATE_WHISPER = 16000
BLOCK_MS = 30
THRESHOLD = 0.01
SILENCIO_MS = 900
MIN_FRASE_MS = 400
MAX_FRASE_MS = 15000

BLOCK_SIZE = int(SAMPLE_RATE_MIC * BLOCK_MS / 1000)
SILENCIO_BLOQUES = int(SILENCIO_MS / BLOCK_MS)
MIN_BLOQUES = int(MIN_FRASE_MS / BLOCK_MS)
MAX_BLOQUES = int(MAX_FRASE_MS / BLOCK_MS)


# ============================================================
# 2. Verificaciones previas
# ============================================================

def verificar_entorno():
    if not os.path.exists(PIPER_MODEL):
        print(f"❌ No encuentro el modelo de voz: {PIPER_MODEL}")
        sys.exit(1)
    if not os.path.exists(os.path.join(BASE_DIR, "fase2_json.py")):
        print("❌ No encuentro 'fase2_json.py' en esta carpeta.")
        sys.exit(1)


# ============================================================
# 3. Cargar Whisper
# ============================================================

print("⏳ Cargando Whisper...")
modelo_whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
print("✅ Whisper cargado.\n")


# ============================================================
# 4. Cola de audio
# ============================================================

cola = queue.Queue()


def callback(indata, frames, time_info, status):
    if status:
        print(f"⚠️ {status}", file=sys.stderr)
    cola.put(indata.copy())


# ============================================================
# 5. Utilidades de audio
# ============================================================

def rms(bloque):
    return float(np.sqrt(np.mean(bloque ** 2)))


def a_whisper(audio):
    audio = audio.flatten().astype(np.float32)
    if SAMPLE_RATE_MIC != SAMPLE_RATE_WHISPER:
        gcd = np.gcd(SAMPLE_RATE_MIC, SAMPLE_RATE_WHISPER)
        audio = resample_poly(audio, SAMPLE_RATE_WHISPER // gcd,
                              SAMPLE_RATE_MIC // gcd).astype(np.float32)
    return audio / (np.max(np.abs(audio)) + 1e-8)


def transcribir(audio):
    segments, _ = modelo_whisper.transcribe(
        audio, language="es", beam_size=5, best_of=5,
        temperature=0.0, vad_filter=False,
        condition_on_previous_text=False
    )
    return " ".join(s.text for s in segments).strip()


# ============================================================
# 6. Generar respuesta textual a partir del JSON
# ============================================================

def generar_respuesta(json_str):
    try:
        data = json.loads(json_str)
    except Exception:
        return "No entendí la orden."

    acciones = data.get("acciones", [])
    if not acciones:
        return "No entendí ninguna orden."

    for acc in acciones:
        if not acc.get("objetos") and not acc.get("personas") and not acc.get("lugares"):
            return "¿Qué quieres que haga?"

    partes = []
    for acc in acciones:
        obj = acc["objetos"][0]["nombre"] if acc.get("objetos") else ""
        adj = " ".join(acc["objetos"][0]["adjetivos"]) if acc.get("objetos") else ""
        lugar = acc["lugares"][0]["nombre"] if acc.get("lugares") else ""
        persona = acc["personas"][0]["nombre"] if acc.get("personas") else ""

        fragmento = f"Voy a {acc['accion']}"
        if adj:
            fragmento += f" {adj}"
        if obj:
            fragmento += f" {obj}"
        if lugar:
            fragmento += f" a {lugar}"
        if persona:
            fragmento += f" a {persona}"
        partes.append(fragmento.strip())

    return " y ".join(partes)


# ============================================================
# 7. Texto a voz
# ============================================================

def sintetizar(texto, salida):
    resultado = subprocess.run(
        [PIPER_EXE, "--model", PIPER_MODEL, "--output_file", salida],
        input=texto.encode("utf-8"),
        capture_output=True,
        check=False
    )
    if resultado.returncode != 0:
        error = resultado.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"Piper falló:\n{error}")
    return salida


def reproducir(ruta):
    """Reproduce el WAV en Windows."""
    try:
        os.startfile(ruta)
    except Exception:
        pass


# ============================================================
# 8. Bucle principal
# ============================================================

def main():
    print("🎤 Habla en español. Ctrl+C para salir.\n")

    buffer = []
    hablando = False
    silencio = 0
    contador = 0

    with sd.InputStream(
        samplerate=SAMPLE_RATE_MIC,
        blocksize=BLOCK_SIZE,
        channels=1,
        dtype="float32",
        callback=callback
    ):
        while True:
            bloque = cola.get()
            energia = rms(bloque)
            hay_voz = energia > THRESHOLD

            if hay_voz:
                if not hablando:
                    hablando = True
                    print("🔴 ", end="", flush=True)
                buffer.append(bloque)
                silencio = 0
            else:
                if hablando:
                    buffer.append(bloque)
                    silencio += 1
                    if silencio >= SILENCIO_BLOQUES:
                        if len(buffer) >= MIN_BLOQUES:
                            audio = np.concatenate(buffer, axis=0)
                            print("\r🟢 Transcribiendo...", end="", flush=True)

                            t0 = time.time()
                            texto = transcribir(a_whisper(audio))
                            t_s2t = time.time() - t0
                            print(f"\r📝 Tú dijiste: {texto}  ({t_s2t:.2f}s)")

                            if texto:
                                contador += 1

                                # --- LLM ---
                                print("🧠 Entendiendo...")
                                t0 = time.time()
                                try:
                                    json_str = extraer_json(texto, LLM_MODEL)
                                    t_llm = time.time() - t0
                                    print(f"📦 JSON: {json_str}  ({t_llm:.2f}s)")
                                except Exception as e:
                                    print(f"⚠️ Error LLM: {e}")
                                    json_str = '{"acciones": []}'
                                    t_llm = 0.0

                                # --- Respuesta textual ---
                                respuesta = generar_respuesta(json_str)
                                print(f"🤖 Robot dice: {respuesta}")

                                # --- T2S ---
                                salida = os.path.join(BASE_DIR, f"respuesta_{contador}.wav")
                                print("🔊 Sintetizando...")
                                t0 = time.time()
                                try:
                                    sintetizar(respuesta, salida)
                                    t_t2s = time.time() - t0
                                    print(f"✅ Audio: {salida}  ({t_t2s:.2f}s)")
                                    reproducir(salida)
                                except Exception as e:
                                    print(f"⚠️ Error en Piper: {e}")
                                    t_t2s = 0.0

                                total = t_s2t + t_llm + t_t2s
                                print(f"⏱️  S2T={t_s2t:.2f}s  LLM={t_llm:.2f}s  T2S={t_t2s:.2f}s  Total={total:.2f}s\n")

                        buffer = []
                        hablando = False
                        silencio = 0

            if hablando and len(buffer) >= MAX_BLOQUES:
                audio = np.concatenate(buffer, axis=0)
                texto = transcribir(a_whisper(audio))
                print(f"\r📝 Tú dijiste: {texto}  (frase larga)\n")
                buffer = []
                hablando = False
                silencio = 0


# ============================================================
# 9. Entry point
# ============================================================

if __name__ == "__main__":
    verificar_entorno()
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Saliendo.")