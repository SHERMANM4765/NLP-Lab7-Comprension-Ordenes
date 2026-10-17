# -*- coding: utf-8 -*-
"""
evaluar.py — Evalúa predicciones contra anotaciones de referencia.

Uso:
    python evaluar.py --predicciones predicciones.json --anotaciones anotaciones.json
"""

import sys
import io
import json
import argparse

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    from jiwer import wer as calcular_wer
    JIWER_OK = True
except ImportError:
    JIWER_OK = False
    print("⚠️  jiwer no instalado. Instálalo con: pip install jiwer\n")


def extraer_entidades(json_obj):
    acciones, objetos, adjetivos, personas, lugares = set(), set(), set(), set(), set()
    for acc in json_obj.get("acciones", []):
        if acc.get("accion"):
            acciones.add(acc["accion"].lower().strip())
        for obj in acc.get("objetos", []):
            if obj.get("nombre"):
                objetos.add(obj["nombre"].lower().strip())
            for adj in obj.get("adjetivos", []):
                adjetivos.add(adj.lower().strip())
        for per in acc.get("personas", []):
            if per.get("nombre"):
                personas.add(per["nombre"].lower().strip())
            for adj in per.get("adjetivos", []):
                adjetivos.add(adj.lower().strip())
        for lug in acc.get("lugares", []):
            if lug.get("nombre"):
                lugares.add(lug["nombre"].lower().strip())
            for adj in lug.get("adjetivos", []):
                adjetivos.add(adj.lower().strip())
    return {"acciones": acciones, "objetos": objetos, "adjetivos": adjetivos,
            "personas": personas, "lugares": lugares}


def f1_set(pred, real):
    if not pred and not real:
        return 1.0
    if not pred or not real:
        return 0.0
    aciertos = len(pred & real)
    precision = aciertos / len(pred) if pred else 0.0
    recall = aciertos / len(real) if real else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def detectar_nivel(nombre_archivo):
    try:
        num = int(nombre_archivo.replace("orden_", "").replace(".wav", ""))
    except Exception:
        return None
    if 1 <= num <= 6: return 1
    if 7 <= num <= 12: return 2
    if 13 <= num <= 18: return 3
    if 19 <= num <= 24: return 4
    if 25 <= num <= 30: return 5
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predicciones", required=True)
    parser.add_argument("--anotaciones", required=True)
    args = parser.parse_args()

    with open(args.predicciones, encoding="utf-8") as f:
        predicciones = json.load(f)
    with open(args.anotaciones, encoding="utf-8") as f:
        anotaciones = json.load(f)

    comunes = sorted(set(predicciones.keys()) & set(anotaciones.keys()))
    if not comunes:
        print("❌ No hay archivos en común.")
        sys.exit(1)

    print(f"📊 Evaluando {len(comunes)} archivos\n")

    f1_cats = {"acciones": [], "objetos": [], "adjetivos": [], "personas": [], "lugares": []}
    f1_por_nivel = {1: [], 2: [], 3: [], 4: [], 5: []}
    f1_global = []
    wer_valores = []
    lat_s2t = []
    lat_llm = []
    json_validos = 0
    detalle = []

    for archivo in comunes:
        pred = predicciones[archivo]
        real = anotaciones[archivo]

        json_pred = pred.get("json", {})
        if isinstance(json_pred, dict) and "acciones" in json_pred:
            json_validos += 1

        pred_ent = extraer_entidades(json_pred)
        real_ent = extraer_entidades(real)

        f1s = []
        for cat in f1_cats:
            f = f1_set(pred_ent[cat], real_ent[cat])
            f1_cats[cat].append(f)
            f1s.append(f)
        f1_arch = sum(f1s) / len(f1s)
        f1_global.append(f1_arch)

        nivel = detectar_nivel(archivo)
        if nivel in f1_por_nivel:
            f1_por_nivel[nivel].append(f1_arch)

        if JIWER_OK:
            ref = real.get("transcripcion", "")
            hip = pred.get("transcripcion", "")
            if ref and hip:
                try:
                    wer_valores.append(calcular_wer(ref.lower(), hip.lower()))
                except Exception:
                    pass

        if "latencia_s2t" in pred:
            lat_s2t.append(pred["latencia_s2t"])
        if "latencia_llm" in pred:
            lat_llm.append(pred["latencia_llm"])

        estado = "✅" if f1_arch >= 0.85 else "⚠️" if f1_arch >= 0.6 else "❌"
        detalle.append((estado, archivo, f1_arch, nivel, pred, real))

    # Reporte
    print("=" * 70)
    print("RESULTADOS GLOBALES")
    print("=" * 70)
    print(f"F1 global promedio:      {sum(f1_global)/len(f1_global):.3f}")
    print(f"JSON válido:             {json_validos}/{len(comunes)} ({100*json_validos/len(comunes):.1f}%)")
    if wer_valores:
        print(f"WER promedio:            {sum(wer_valores)/len(wer_valores):.2%}")
    if lat_s2t:
        print(f"Latencia S2T (media):    {sum(lat_s2t)/len(lat_s2t):.2f}s")
    if lat_llm:
        print(f"Latencia LLM (media):    {sum(lat_llm)/len(lat_llm):.2f}s")
    if lat_s2t and lat_llm:
        total = (sum(lat_s2t)/len(lat_s2t)) + (sum(lat_llm)/len(lat_llm))
        print(f"Latencia total (media):  {total:.2f}s")

    print()
    print("=" * 70)
    print("F1 POR CATEGORÍA")
    print("=" * 70)
    for cat, valores in f1_cats.items():
        if valores:
            print(f"  {cat:<12} {sum(valores)/len(valores):.3f}")

    print()
    print("=" * 70)
    print("F1 POR NIVEL DE DIFICULTAD")
    print("=" * 70)
    for nivel in sorted(f1_por_nivel):
        vals = f1_por_nivel[nivel]
        if vals:
            print(f"  Nivel {nivel}      {sum(vals)/len(vals):.3f}  ({len(vals)} archivos)")

    print()
    print("=" * 70)
    print("DETALLE POR ARCHIVO")
    print("=" * 70)
    for estado, archivo, f1_arch, nivel, pred, real in detalle:
        nivel_str = f"N{nivel}" if nivel else "??"
        print(f"{estado} {archivo:<20} [{nivel_str}]  F1={f1_arch:.2f}")
        print(f"      📝 Pred: {pred.get('transcripcion', '')[:70]}")
        print(f"      ✅ Real: {real.get('transcripcion', '')[:70]}")
        print()

    resumen = {
        "num_archivos": len(comunes),
        "f1_global": sum(f1_global) / len(f1_global),
        "f1_por_categoria": {c: sum(v) / len(v) for c, v in f1_cats.items() if v},
        "f1_por_nivel": {n: sum(v) / len(v) for n, v in f1_por_nivel.items() if v},
        "json_validos": json_validos,
        "json_validos_pct": 100 * json_validos / len(comunes),
        "wer_promedio": sum(wer_valores) / len(wer_valores) if wer_valores else None,
        "latencia_s2t_media": sum(lat_s2t) / len(lat_s2t) if lat_s2t else None,
        "latencia_llm_media": sum(lat_llm) / len(lat_llm) if lat_llm else None,
    }

    with open("evaluacion.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("💾 Resultados guardados en: evaluacion.json")
    print("=" * 70)


if __name__ == "__main__":
    main()