# Laboratorio 7: Comprensión de Órdenes en Español para Robots con LLMs Ligeros

**Asignatura:** NLP y LLMs  
**Integrantes:** Santiago Lopez, Alejandra Avila y Nicolas Reyes
**Fecha:** Octubre 2026

## 📌 1. Descripción General
Este repositorio contiene la solución completa al Laboratorio 7. Se implementó un sistema de extremo a extremo capaz de capturar órdenes habladas en español, transcribirlas, extraer entidades en un JSON estructurado y devolver una respuesta hablada. 

El sistema cumple estrictamente con las restricciones del laboratorio:
*   **100% Local:** Sin conexión a internet durante la inferencia.
*   **Hardware:** Solo CPU (sin GPU), consumiendo menos de 8 GB de RAM.
*   **Modelo:** Qwen2.5:1.5B (menos de 4 mil millones de parámetros).
*   **Formato:** Salida JSON estricta y validada.

## 🏗️ 2. Arquitectura del Pipeline
1.  **Voz a Texto (S2T):** `faster-whisper` (modelo `small` con cuantización `int8`).
2.  **Comprensión (LLM):** `Qwen2.5:1.5B` ejecutado con Ollama. Se utiliza **Pydantic** y salida restringida para forzar el JSON, más **spaCy** para lematización.
3.  **Texto a Voz (T2S):** `Piper TTS` (voz `es_ES-davefx-medium`).

## 🛠️ 3. Instalación y Requisitos Previos

Antes de ejecutar cualquier script, asegúrese de tener el entorno configurado:

1. **Instalar Ollama y descargar el modelo:**
   ```bash
   ollama pull qwen2.5:1.5b
