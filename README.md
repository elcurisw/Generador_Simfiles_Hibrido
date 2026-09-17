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

```bash
py -3.12 -m pip install librosa customtkinter
```

Para generar archivos ejecutables (.exe), instala pyinstaller:

```bash
py -3.12 -m pip install pyinstaller
```

⚡ ¡ATENCIÓN! VERSIONES CRÍTICAS DE TORCH (Para evitar conflictos con DLL en Windows)

```bash
py -3.12 -m pip install torch==2.8.0 torchvision==0.23.0
```

Para el creador de checkpoints, instalar tqdm:

```bash
py -3.12 -m pip install tqdm
```

### B. Flujo de Trabajo (Dos Etapas)

El proceso consta de dos fases obligatorias: 1) Crear los Checkpoints del modelo y 2) Generar los pasos con el generador principal.

#### Fase 1: Creación de Checkpoints IA (Obligatorio)

Antes de usar el generador, debes crear el modelo base utilizando la siguiente ruta. Recuerda cargar tus archivos .sm con música en carpetas dentro del directorio StepMania_Songs_Pack.

```bash
py -3.12 stepmania_pipeline.py
```

El archivo generado será guardado en una nueva carpeta llamada checkpoints, renombralo como step_transformer_model si deseas usar la función automática del generador.

#### Fase 2: Ejecución del Generador de Pasos

Una vez generados los checkpoints, puedes correr el generador principal. Por defecto, buscará un modelo llamado step_transformer_model.pt en la carpeta checkpoints, es opcional.

```bash
py -3.12 generador_simfiles_hibrido.py
```

## ⚙️ II. Interfaz y Parámetros de Configuración (Los 45 Controles)

La interfaz está dividida lógicamente para facilitar la generación, desde las entradas principales hasta los ajustes más detallados de ritmo y IA.

### A. Inputs Esenciales y Meta-Datos

Estos campos definen qué se va a generar.

| Nº | Campo | Tipo | Descripción | Notas Clave |
| :---: | :---: | :---: | :---: | :---: |
| 1 |	Seleccionar Canción | Selector de Audio |	Carga el archivo de audio que servirá como base para la creación de pasos. | Obligatorio. |
| 2	| Seleccionar Checkpoint IA |	Selector de Archivo |	Carga el modelo (checkpoint) generado en la Fase 1.	| Obligatorio en esta versión del software. |
| 3	| Título de la canción | Texto | Nombre que se le asignará a tu pieza musical. Se usa para renombrar los archivos. | Sugerencia: Mantenerlo conciso. |
| 4	| Renombrar archivos | Checkbox/Campo Op. |	Permite forzar el nombre del archivo de pasos con el título proporcionado. | Útil para organizar el pack final. |

### B. Control Temporal y Estructura (Timing)

Definen la duración, velocidad base y ajustes temporales de la canción.

| Nº | Campo | Tipo | Descripción | Notas Clave |
| :---: | :---: | :---: | :---: | :---: |
| 5 |	Duración Máxima | Texto | Define el tiempo máximo que debe tener el archivo de pasos generado. |	Útil para recortar o limitar la extensión. |
| 6 |	Configuración BPM |	Control Numérico | Establece los pulsos por minuto (BPM) deseados. Se pueden ajustar con botones incrementales (1 en 1). | Define el ritmo base de la canción. |
| 7 |	Doble BPM | Checkbox | Si el resultado automático o manual no es satisfactorio, puedes duplicar el BPM aquí antes de generar pasos. | Ajuste avanzado de ritmo. |
| 8 |	Aplicar BPM Dinámico | Checkbox/Restrictivo | Activa un ajuste que varía el BPM a lo largo de la canción según las fluctuaciones rítmicas del audio. | 🛑 Si activas esto, desactiva "Aplicar Velocidad Visual". |
| 9 |	Aplicar Velocidad Visual | Checkbox/Restrictivo |	Aplica efectos visuales basados en los niveles de ritmo (velocidad) detectados en la canción. |	🛑 Si activas esto, desactiva "Aplicar BPM Dinámico". |
| 15 | Offset de Inicio | Selector | Permite definir manualmente el punto exacto donde debe comenzar la generación de pasos, distinto a la duración total. | Útil si el inicio es silencioso o no rítmico. |
| 16 | Detectar Offset Automáticamente | Checkbox | Si está desactivado, debes especificar un offset manual (punto 15). | Se recomienda deshabilitar si se conoce el punto de inicio preciso. |
| 17 | Extensión Final Estética | Control Numérico | Permite extender el audio sin generar pasos rítmicos adicionales. | Ideal para escuchar la "desvanencia" final del track. |

