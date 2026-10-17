# -*- coding: utf-8 -*-
"""
Genera un borrador de anotaciones a partir de predicciones.json.
Después hay que revisarlo y corregirlo a mano.
"""
import json

with open("predicciones.json", encoding="utf-8") as f:
    predicciones = json.load(f)

anotaciones = {}
for archivo, pred in predicciones.items():
    anotaciones[archivo] = {
        "transcripcion": pred.get("transcripcion", ""),
        "acciones": pred.get("json", {}).get("acciones", [])
    }

with open("anotaciones.json", "w", encoding="utf-8") as f:
    json.dump(anotaciones, f, ensure_ascii=False, indent=2)

print(f"✅ anotaciones.json generado con {len(anotaciones)} entradas")
print("⚠️  Ahora ábrelo y corrige los errores a mano.")