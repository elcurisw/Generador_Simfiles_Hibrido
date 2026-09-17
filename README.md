# Generador_Simfiles_Hibrido

## ⚠️ ADVERTENCIA DE SEGURIDAD CRÍTICA (IMPORTANTE)
Este software trabaja directamente con la generación y modificación de archivos asociados.

Riesgo de Archivo: Dado que el programa está diseñado para crear, modificar y reescribir contenido rítmico basado en algoritmos avanzados, existe un riesgo inherente de "Falsos Positivos" de seguridad por el antivirus. Usa este programa considerando hacer respaldos de tus archivos.

## 🎶Guía de Usuario: Generador de Pasos Híbrido (StepMania)

Este programa está diseñado para generar archivos de pasos (*.sm y *.ssc) dinámicos y detallados, optimizados para juegos de ritmo como StepMania. Sigue esta guía para garantizar una instalación correcta y la mejor experiencia de uso posible.

### I. Requisitos Previos e Instalación Técnica

⚠️ Recomendaciones Importantes:

Se recomienda utilizar un entorno virtual (venv) para aislar las dependencias del proyecto, ya sea para probar el código o entrenar modelos.
El entorno base de prueba es Python 3.12 en WINDOWS 11, pero se espera compatibilidad con otras versiones de Python o Windows. 

**La versión ejecutable (.exe) con un checkpoint aparte se encuentra disponible si solo desea utilizar el generador.**

#### A. Dependencias Principales (Instalación vía pip)

Instalar dependencias esenciales

\```bash

py -3.12 -m pip install librosa customtkinter

\```

Para generar archivos ejecutables (.exe), instala pyinstaller:

`py -3.12 -m pip install pyinstaller`

⚡ ¡ATENCIÓN! VERSIONES CRÍTICAS DE TORCH (Para evitar conflictos con DLL en Windows)

`py -3.12 -m pip install torch==2.8.0 torchvision==0.23.0`

Para el creador de checkpoints, instalar tqdm:

`py -3.12 -m pip install tqdm`

### B. Flujo de Trabajo (Dos Etapas)

El proceso consta de dos fases obligatorias: 1) Crear los Checkpoints del modelo y 2) Generar los pasos con el generador principal.

#### Fase 1: Creación de Checkpoints IA (Obligatorio)

Antes de usar el generador, debes crear el modelo base utilizando la siguiente ruta. Recuerda cargar tus archivos .sm con música en carpetas dentro del directorio StepMania_Songs_Pack.

`py -3.12 stepmania_pipeline.py`


