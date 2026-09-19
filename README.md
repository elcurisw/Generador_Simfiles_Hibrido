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
py -3.12 -m pip install librosa customtkinter matplotlib
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
| 5 | Nombre del artista | Text | Nombra el artista de la canción | Opcional |

### B. Control Temporal y Estructura (Timing)

Definen la duración, velocidad base y ajustes temporales de la canción.

| Nº | Campo | Tipo | Descripción | Notas Clave |
| :---: | :---: | :---: | :---: | :---: |
| 6 |	Duración Máxima + Ajustar Límites en Gráfica Interactiva | Texto + Botón | Define el tiempo máximo que debe tener el archivo de pasos generado. El botón te abre una interfaz para colocar las zonas de duración, offset y extensión. |	Útil para recortar o limitar la extensión. |
| 7 |	Configuración BPM |	Control Numérico | Establece los pulsos por minuto (BPM) deseados. Se pueden ajustar con botones incrementales (1 en 1). | Define el ritmo base de la canción. |
| 8 |	Doble BPM | Checkbox | Si el resultado automático o manual no es satisfactorio, puedes duplicar el BPM aquí antes de generar pasos. | Ajuste avanzado de ritmo. |
| 9 |	Aplicar BPM Dinámico | Checkbox/Restrictivo | Activa un ajuste que varía el BPM a lo largo de la canción según las fluctuaciones rítmicas del audio. | 🛑 Si activas esto, desactiva "Aplicar Velocidad Visual". |
| 10 |	Aplicar Velocidad Visual | Checkbox/Restrictivo |	Aplica efectos visuales basados en los niveles de ritmo (velocidad) detectados en la canción. |	🛑 Si activas esto, desactiva "Aplicar BPM Dinámico". |
| 16 | Offset de Inicio | Selector | Permite definir manualmente el punto exacto donde debe comenzar la generación de pasos, distinto a la duración total. | Útil si el inicio es silencioso o no rítmico. |
| 17 | Detectar Offset Automáticamente | Checkbox | Si está desactivado, debes especificar un offset manual (punto 15). | Se recomienda deshabilitar si se conoce el punto de inicio preciso. |
| 18 | Extensión Final Estética | Control Numérico | Permite extender el audio sin generar pasos rítmicos adicionales. | Ideal para escuchar la "desvanencia" final del track. |

### C. Análisis Rítmico Avanzado (Sensibilidad y Ritmo)

Estos parámetros refinan cómo el sistema interpreta el pulso y la energía de la canción.

| Nº | Campo | Tipo | Descripción | Notas Clave |
| :---: | :---: | :---: | :---: | :---: |
| 11 | Scroll mínimo (Low Sensitivity) | Control Numérico |	Define qué nivel o ritmo se considera una sección lenta o "calmada". En estas zonas aparecerán menos notas. |	Ajusta la sensibilidad a partes suaves. |
| 12 | Scroll Máximo (High Sensitivity) |	Control Numérico | Define qué nivel o ritmo se considera una sección rápida o intensa. Aparecerá mayor densidad de notas. |	Ajusta el umbral para picos rítmicos fuertes. |
| 13 | Duración transición | Control Numérico |	Determina cuánto tiempo tardará la herramienta en reajustar la velocidad cuando se usa "Aplicar Velocidad Visual". | Controla la suavidad del cambio de ritmo. |
| 40 | Sensibilidad RMS Mínimo | Control Numérico |	Establece el parámetro mínimo (RMS) que debe tener una sección para ser considerada rítmicamente notable por la IA. |	Avanzado. Impacto en secciones suaves. |
| 41 | Sensibilidad RMS Máximo | Control Numérico |	Establece el parámetro máximo (RMS) que define el pico de impacto rítmico más fuerte de la canción.	| Avanzado. Impacto en picos intensos. |

### D. Ajustes Estilísticos y Procesamiento AI

Controles para mejorar, modificar o controlar la "personalidad" del ritmo generado por IA.

| Nº | Campo | Tipo | Descripción | Notas Clave |
| :---: | :---: | :---: | :---: | :---: |
| 14 | Aplicar Posprocesamiento rítmico |	Checkbox/Op. | Mejora el resultado bruto de la IA mediante efectos adicionales. Desactivarlo produce un resultado "en crudo". |	Recomendado activarlo para mejor calidad. |
| 15 | Recalcular Dificultad Dinámicamente | Checkbox/Op.	| Permite recalcular la dificultad basándose en las elecciones manuales de usuario (ej: BPM, Scroll Min/Max). |	Si se desactiva, se usa una configuración fija. |
| 21 | Temperatura IA |	Control Numérico | Determina cuán "libre" o creativa puede ser la Inteligencia Artificial al generar los pasos. |	Un valor alto = más experimentación; bajo = más conservador y predecible. |
| 42 | Dificultad Techo del Pack | Control Numérico |	Define el nivel general de dificultad deseado para todo el pack de pasos. Se usa en el recálculo de dificultad (Punto 14). | Establece la intención artística del resultado final. |
| 38 | Potenciar Efectos y Trampas | Checkbox/Op. |	Activar este campo aplica efectos especiales avanzados a los pasos generados por IA. | Uso avanzado, mejora el realismo rítmico. |

### E. Control de Elementos Rítmicos Especiales (Minas, Fakes, etc.)

Permiten añadir elementos específicos del género de juegos de ritmo para aumentar la complejidad y la variación.

| Nº | Campo | Tipo | Descripción | Notas Clave |
| :---: | :---: | :---: | :---: | :---: |
| 22 | Duración Máxima de Holds |	Control Numérico | Define el tiempo máximo que puede mantener un paso sostenido (Hold).	| Controla la duración de las notas largas. |
| 23 | Máximo de Holds Simultáneos | Control Numérico |	Determina cuántos pasos sostenidos pueden ocurrir al mismo tiempo. | Ideal para quitar o aumentar complejidad en zonas específicas. |
| 24 - 37 |	(Minas, Fakes, Lifts, Potions, Shields, Rayos, Hiddens) | Probabilidad / Máximo Control Numérico | Cada uno de estos grupos controla la probabilidad y el número máximo de un efecto específico por compás generado. | Estos son ajustes muy finos para crear patrones específicos (ej: si quieres muchas trampas/minas). |

### F. Controles Finales y Salida

Ajustes finales antes de ejecutar la generación.

| Nº | Campo | Tipo | Descripción | Notas Clave |
| :---: | :---: | :---: | :---: | :---: |
| 43 | Semilla de Generación (Seed)	| Número | Si introduces un número, el generador intentará crear exactamente los mismos pasos. | Guardar la semilla es útil para reproducir resultados perfectos o deseados. |
| 44 | Restablecer Parámetros |	Botón | Limpia todos los campos y devuelve a sus valores predeterminados (por default).	| Recomendado si estás empezando desde cero. |
| 19, 20 | Banner Graphic / Video de Fondo | Selector Multimedia | Elementos opcionales para la presentación visual del track. | No afectan la generación de pasos, solo el output multimedia. |
| 45 | Procesar y Exportar Dual Pack | Botón de Ejecución | Ejecuta todas las configuraciones establecidas para generar el archivo final de pasos (.sm, .ssc). | Acción principal al finalizar la configuración. |
| 46 | Monitor de Densidad en Tiempo Real | Consola | Proporciona una interpretación aproximada e inmediata del resultado rítmico que se está generando. | Sirve como un feedback visual mientras ajustas los parámetros. |
