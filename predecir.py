# -*- coding: utf-8 -*-
"""
predecir.py — Sistema de comprensión de órdenes por lotes.

Uso:
    python predecir.py --entrada audios/ --salida predicciones.json

Recorre todos los audios de la carpeta de entrada, aplica el pipeline
S2T + LLM + post-procesamiento, y guarda todas las predicciones en JSON.
"""

import sys
import io
import os
import json
import time
import argparse

from faster_whisper import WhisperModel

# UTF-8 en consola
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from fase2_json import extraer_json, Orden, validar_json


# ============================================================
# Configuración
# ============================================================

WHISPER_MODEL = "small"
LLM_MODEL = "qwen2.5:1.5b"
EXTENSIONES_AUDIO = {".wav", ".mp3", ".m4a", ".mp4", ".ogg", ".flac"}


# ============================================================
# Cargar Whisper una sola vez
# ============================================================

print("⏳ Cargando Whisper...")
_modelo_whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
print("✅ Whisper cargado.\n")


def transcribir(audio_path):
    segments, _ = _modelo_whisper.transcribe(
        audio_path,
        language="es",
        beam_size=5,
        best_of=5,
        temperature=0.0,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False
    )
    return " ".join(s.text for s in segments).strip()


# ============================================================
# Pipeline por archivo
# ============================================================

def procesar_audio(audio_path):
    """Devuelve un dict con transcripción, JSON y latencias."""
    resultado = {
        "archivo": os.path.basename(audio_path),
        "transcripcion": "",
        "json": {"acciones": []},
        "error": None,
        "latencia_s2t": 0.0,
        "latencia_llm": 0.0,
    }

    # S2T
    try:
        t0 = time.time()
        texto = transcribir(audio_path)
        resultado["latencia_s2t"] = time.time() - t0
        resultado["transcripcion"] = texto
    except Exception as e:
        resultado["error"] = f"S2T: {e}"
        return resultado

    if not texto:
        return resultado

    # LLM
    try:
        t0 = time.time()
        json_str = extraer_json(texto, LLM_MODEL)
        resultado["latencia_llm"] = time.time() - t0
        resultado["json"] = json.loads(json_str)
    except Exception as e:
        resultado["error"] = f"LLM: {e}"

    return resultado


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sistema de comprensión de órdenes por lotes."
    )
    parser.add_argument("--entrada", required=True,
                        help="Carpeta con los audios (ej. audios/)")
    parser.add_argument("--salida", required=True,
                        help="Archivo JSON de salida (ej. predicciones.json)")
    args = parser.parse_args()

    if not os.path.isdir(args.entrada):
        print(f"❌ No existe la carpeta: {args.entrada}")
        sys.exit(1)

    # Listar audios
    archivos = []
    for nombre in sorted(os.listdir(args.entrada)):
        ruta = os.path.join(args.entrada, nombre)
        ext = os.path.splitext(nombre)[1].lower()
        if os.path.isfile(ruta) and ext in EXTENSIONES_AUDIO:
            archivos.append(ruta)

    if not archivos:
        print(f"❌ No se encontraron audios en {args.entrada}")
        sys.exit(1)

    print(f"📂 Procesando {len(archivos)} audios desde {args.entrada}\n")

    predicciones = {}
    t_inicio = time.time()

    for i, ruta in enumerate(archivos, 1):
        nombre = os.path.basename(ruta)
        print(f"[{i}/{len(archivos)}] {nombre}")
        r = procesar_audio(ruta)
        predicciones[nombre] = r

        if r["error"]:
            print(f"   ❌ {r['error']}")
        else:
            print(f"   📝 {r['transcripcion']}")
            print(f"   📦 {json.dumps(r['json'], ensure_ascii=False)}")
        print()

    t_total = time.time() - t_inicio

    # Guardar
    with open(args.salida, "w", encoding="utf-8") as f:
        json.dump(predicciones, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print(f"✅ Predicciones guardadas en: {args.salida}")
    print(f"⏱️  Tiempo total: {t_total:.2f}s ({t_total/len(archivos):.2f}s por audio)")
    print(f"📊 Audios procesados: {len(archivos)}")


if __name__ == "__main__":
    main()