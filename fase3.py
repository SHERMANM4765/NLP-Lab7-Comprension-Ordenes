# -*- coding: utf-8 -*-
import sys
import time
import json
import psutil
import os
from fase2_json import extraer_json

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

MODELOS = ["qwen2.5:1.5b", "gemma2:2b", "phi3:mini"]
ORDENES = ["Trae el vaso", "Pon el libro azul sobre la mesa", "Tráeme eso"]

def evaluar_modelo(modelo):
    latencias = []
    ram_antes = psutil.Process(os.getpid()).memory_info().rss / 1e9
    
    for orden in ORDENES:
        inicio = time.time()
        try:
            json_str = extraer_json(orden, modelo=modelo)
            json.loads(json_str)
        except Exception:
            pass
        latencias.append(time.time() - inicio)
        
    ram_despues = psutil.Process(os.getpid()).memory_info().rss / 1e9
    latencia_promedio = sum(latencias) / len(latencias)
    
    return {"modelo": modelo, "latencia": latencia_promedio, "ram": ram_despues - ram_antes}

if __name__ == "__main__":
    print(f"{'Modelo':<20} | {'Latencia Prom':<15} | {'RAM Delta'}")
    print("-" * 55)
    for m in MODELOS:
        r = evaluar_modelo(m)
        print(f"{r['modelo']:<20} | {r['latencia']:.2f} segundos  | {r['ram']:.2f} GB")