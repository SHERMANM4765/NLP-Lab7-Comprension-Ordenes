# -*- coding: utf-8 -*-
import sys
import json
import re
from typing import List
import ollama
from pydantic import BaseModel

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# --- 1. Configurar spaCy ---
try:
    import spacy
    _NLP = spacy.load("es_core_news_sm")
except Exception:
    _NLP = None
    print("⚠️ spaCy no está instalado. Usando reglas básicas.")

# --- 2. Esquema Pydantic ---
class Objeto(BaseModel):
    nombre: str
    adjetivos: List[str] = []

class Persona(BaseModel):
    nombre: str
    adjetivos: List[str] = []

class Lugar(BaseModel):
    nombre: str
    adjetivos: List[str] = []

class Accion(BaseModel):
    accion: str
    objetos: List[Objeto] = []
    personas: List[Persona] = []
    lugares: List[Lugar] = []

class Orden(BaseModel):
    acciones: List[Accion] = []

# --- 3. Prompt Few-Shot ---
PROMPT_SISTEMA = """Eres un extractor de información de órdenes en español.
Devuelve ÚNICAMENTE un JSON válido.
REGLAS:
1. Acción en infinitivo y minúscula ("tráeme" -> "traer").
2. Nombres sin artículo.
3. Pronombres sin referente ("eso") no se anotan.
4. "tú" y "yo" no se anotan como personas.

EJEMPLOS:
Orden: "Trae el vaso"
Salida: {"acciones": [{"accion": "traer", "objetos": [{"nombre": "vaso", "adjetivos": []}], "personas": [], "lugares": []}]}

Orden: "Tráeme eso"
Salida: {"acciones": []}
"""

# --- 4. Lematización (spaCy) ---
def _lematizar_verbo(texto):
    if not texto: return texto
    texto = texto.strip().lower()
    base = re.sub(r"(me|te|le|lo|la|los|las|nos|se|sela|selo)$", "", texto)
    if base.endswith(("ar", "er", "ir")): return base
    
    if _NLP:
        doc = _NLP(base)
        for token in doc:
            if token.pos_ == "VERB":
                return token.lemma_.lower()
    
    reglas = {"dile": "decir", "trae": "traer", "lleva": "llevar", "ve": "ir"}
    return reglas.get(base, base)

# --- 5. Post-procesamiento ---
PRONOMBRES_VACIOS = {"eso", "esto", "aquello", "algo", "nada"}

def _postprocesar(orden: Orden) -> Orden:
    acciones_limpias = []
    for acc in orden.acciones:
        acc.accion = _lematizar_verbo(acc.accion)
        acc.objetos = [o for o in acc.objetos if o.nombre.lower() not in PRONOMBRES_VACIOS]
        
        if acc.objetos or acc.personas or acc.lugares or acc.accion in ["ir", "saludar"]:
            acciones_limpias.append(acc)
            
    return Orden(acciones=acciones_limpias)

# --- 6. Función Principal ---
def extraer_json(texto_orden, modelo="qwen2.5:1.5b"):
    if not texto_orden: return '{"acciones": []}'
    
    respuesta = ollama.chat(
        model=modelo,
        messages=[
            {"role": "system", "content": PROMPT_SISTEMA},
            {"role": "user", "content": f'Orden: "{texto_orden}"'}
        ],
        format=Orden.model_json_schema(),
        options={"temperature": 0}
    )
    
    obj = Orden.model_validate_json(respuesta["message"]["content"])
    obj_limpio = _postprocesar(obj)
    return obj_limpio.model_dump_json()