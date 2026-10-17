# -*- coding: utf-8 -*-
"""
Genera 30 audios con órdenes reales usando Piper.
Cada audio contiene una orden, NO una respuesta.
"""
import os
import subprocess
import shutil

PIPER_MODEL = "voces/es_ES-davefx-medium.onnx"
PIPER_EXE = shutil.which("piper") or "piper"

ORDENES = [
    # Nivel 1 - Simple (6)
    "Trae el vaso",
    "Saluda al visitante",
    "Busca el libro",
    "Lleva la taza",
    "Toma el paquete",
    "Pon el celular",

    # Nivel 2 - Descriptiva (6)
    "Pon el libro azul sobre la mesa pequeña",
    "Lleva el paquete grande a la oficina principal",
    "Trae la taza roja de la cocina",
    "Busca el celular negro en el cuarto",
    "Lleva la cobija gris a la sala",
    "Pon el vaso verde sobre la mesa grande",

    # Nivel 3 - Compuesta (6)
    "Ve al cuarto, toma la cobija gris y tráesela a la abuela",
    "Recibe a la señora Gómez, acompáñala a la sala de reuniones y avísale a Carlos",
    "Busca el libro azul, llévalo a la oficina y entrégaselo a María",
    "Ve a la cocina, toma la taza roja y tráemela",
    "Recibe al señor Pérez y acompáñalo a la sala",
    "Toma el paquete grande y llévalo a la oficina principal",

    # Nivel 4 - Coloquial (6)
    "Oye porfa me alcanzas el celular que dejé en el cuarto",
    "Eh mira dile al señor de la corbata roja que ya lo atienden",
    "Oye necesito que me traigas la cobija que está en el cuarto",
    "Porfa lleva esto a la oficina y dale esto a Juan",
    "Mira, tráeme el vaso de la cocina cuando puedas",
    "Oye, ¿me ayudas llevando este paquete a la sala?",

    # Nivel 5 - Trampa (6)
    "Tráeme eso",
    "Qué demora la de hoy",
    "Ahí está",
    "Ya te dije",
    "Otra vez",
    "Eso mismo",
]

os.makedirs("audios", exist_ok=True)

for i, orden in enumerate(ORDENES, 1):
    salida = f"audios/orden_{i:03d}.wav"
    subprocess.run(
        [PIPER_EXE, "--model", PIPER_MODEL, "--output_file", salida],
        input=orden.encode("utf-8"),
        check=True
    )
    print(f"✅ {salida}  →  {orden}")

print(f"\n🎧 {len(ORDENES)} audios generados en 'audios/'")