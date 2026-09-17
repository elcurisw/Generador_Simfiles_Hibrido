# Generador_Simfiles_Hibrido

## ⚠️ CRITICAL SECURITY WARNING (IMPORTANT)
This software works directly with the generation and modification of associated files.

File Risk: Since the program is designed to create, modify, and rewrite rhythmic content based on advanced algorithms, there is an inherent risk of "False Positives" from antivirus software. Use this program considering making backups of your files.

## 🎶User Guide: Hybrid Step Generator (StepMania)

This program is designed to generate dynamic and detailed step files (*.sm and *.ssc), optimized for rhythm games like StepMania. Follow this guide to ensure correct installation and the best possible user experience.

### I. Prerequisites and Technical Installation

⚠️ Important Recommendations:

It is recommended to use a virtual environment (venv) to isolate project dependencies, whether testing code or training models.
The base test environment is Python 3.12 on WINDOWS 11, but compatibility with other versions of Python or Windows is expected.

**The executable version (.exe) with a separate checkpoint is available if you only wish to use the generator.**

#### A. Main Dependencies (Installation via pip)

Install essential dependencies

```bash
py -3.12 -m pip install librosa customtkinter
```

To generate executables (.exe), install pyinstaller:

```bash
py -3.12 -m pip install pyinstaller
```

⚡ ATTENTION! CRITICAL TORCH VERSIONS (To avoid DLL conflicts on Windows)

```bash
py -3.12 -m pip install torch==2.8.0 torchvision==0.23.0
```

For the checkpoint creator, install tqdm:

```bash
py -3.12 -m pip install tqdm
```

### B. Workflow (Two Stages)

The process consists of two mandatory phases: 1) Creating Model Checkpoints and 2) Generating Steps with the main generator.

#### Phase 1: IA Checkpoint Creation (Mandatory)

Before using the generator, you must create the base model using the following path. Remember to load your .sm files with music into folders within the StepMania_Songs_Pack directory.

```bash
py -3.12 stepmania_pipeline.py
```

The generated file will be saved in a new folder called checkpoints; rename it to step_transformer_model if you want to use the generator's automatic function.

#### Phase 2: Step Generator Execution

Once checkpoints are generated, you can run the main generator. By default, it will look for a model named step_transformer_model.pt in the checkpoints folder (optional).

```bash
py -3.12 generador_simfiles_hibrido.py
```

## ⚙️ II. Interface and Configuration Parameters (The 45 Controls)

The interface is logically divided to facilitate generation, from the main inputs to the most detailed rhythm and AI adjustments.

### A. Essential Inputs and Meta-Data

These fields define what will be generated.

| Nº | Field | Type | Description | Key Notes |
| :---: | :---: | :---: | :---: | :---: |
| 1 |	Seleccionar Canción | Audio Selector | Loads the audio file that will serve as the basis for step creation. | Mandatory. |
| 2	| Seleccionar Checkpoint IA | File Selector | Loads the model (checkpoint) generated in Phase 1. | Mandatory in this version of the software. |
| 3	| Título de la canción | Text | The name assigned to your musical piece. Used to rename files. | Suggestion: Keep it concise. |
| 4	| Renombrar archivos | Checkbox/Opt Field |	Allows forcing the step file name with the provided title. | Useful for organizing the final pack. |

### B. Temporal and Structure Control (Timing)

Define the duration, base speed, and temporal adjustments of the song.

| Nº | Field | Type | Description | Key Notes |
| :---: | :---: | :---: | :---: | :---: |
| 5 |	Duración Máxima | Text | Defines the maximum time the generated step file should have. |	Useful for trimming or limiting the length. |
| 6 |	Configuración BPM |	Numeric Control | Sets the desired beats per minute (BPM). Can be adjusted with incremental buttons (by 1). | Defines the song's base rhythm. |
| 7 |	Doble BPM | Checkbox | If the automatic or manual result is not satisfactory, you can double the BPM here before generating steps. | Advanced rhythm adjustment. |
| 8 |	Aplicar BPM Dinámico | Checkbox/Restrictive | Activates an adjustment that varies the BPM throughout the song according to rhythmic fluctuations in the audio. | 🛑 If you activate this, disable "Aplicar Velocidad Visual". |
| 9 |	Aplicar Velocidad Visual | Checkbox/Restrictive	| Applies visual effects based on rhythm levels (speed) detected in the song. |	🛑 If you activate this, disable "Aplicar BPM Dinámico". |
| 15 | Offset de Inicio | Selector | Allows manually defining the exact point where step generation must start, separate from the total duration. | Useful if the beginning is silent or non-rhythmic. |
| 16 | Detectar Offset Automáticamente | Checkbox | If disabled, you must specify a manual offset (point 15). | Recommended to disable if the precise starting point is known. |
| 17 | Extensión Final Estética | Numeric Control | Allows extending the audio without generating additional rhythmic steps. | Ideal for listening to the final "fade-out" of the track. |

### C. Advanced Rhythm Analysis (Sensitivity and Rhythm)

These parameters refine how the system interprets the pulse and energy of the song.

| Nº | Field | Type | Description | Key Notes |
| :---: | :---: | :---: | :---: | :---: |
| 10 | Scroll mínimo (Low Sensitivity) | Numeric Control |	Defines what level or rhythm is considered a slow or "calm" section. Fewer notes will appear in these areas. | Adjusts sensitivity for softer parts. |
| 11 | Scroll Máximo (High Sensitivity) | Numeric Control | Defines what level or rhythm is considered a fast or intense section. A higher density of notes will appear. | Adjusts the threshold for strong rhythmic peaks. |
| 12 | Duración transición | Numeric Control | Determines how long the tool will take to readjust speed when using "Apply Visual Speed." | Controls the smoothness of rhythm changes. |
| 39 | Sensibilidad RMS Mínimo | Numeric Control |	Establishes the minimum parameter (RMS) that a section must have to be considered rhythmically notable by the AI. |	Advanced. Impact on soft sections. |
| 40 | Sensibilidad RMS Máximo | Numeric Control |	Establishes the maximum parameter (RMS) defining the strongest rhythmic impact peak of the song.	| Advanced. Impact on intense peaks. |

### D. Stylistic Adjustments and AI Processing

Controls to improve, modify, or control the "personality" of the rhythm generated by AI.

| Nº | Field | Type | Description | Key Notes |
| :---: | :---: | :---: | :---: | :---: |
| 13 | Aplicar Posprocesamiento rítmico | Checkbox/Opt. | Improves the raw output from the AI using additional effects. Disabling it produces a "raw" result. | Recommended to activate for better quality. |
| 14 | Recalcular Dificultad Dinámicamente | Checkbox/Opt. | Allows recalculating difficulty based on user's manual choices (e.g., BPM, Scroll Min/Max). | If disabled, uses a fixed configuration. |
| 20 | Temperatura IA |	Numeric Control | Determines how "free" or creative the Artificial Intelligence can be when generating steps. |	High value = more experimentation; low = more conservative and predictable. |
| 41 | Dificultad Techo del Pack | Numeric Control | Defines the general desired difficulty level for the entire step pack. Used in difficulty recalculation (Point 14). | Establishes the artistic intent of the final result. |
| 37 | Potenciar Efectos y Trampas | Checkbox/Opt. |	Activating this field applies advanced special effects to the steps generated by AI. | Advanced usage, improves rhythmic realism. |

### E. Control of Special Rhythm Elements (Minas, Fakes, etc.)

Allow adding specific elements from the rhythm game genre to increase complexity and variation.

| Nº | Field | Type | Description | Key Notes |
| :---: | :---: | :---: | :---: | :---: |
| 21 | Duración Máxima de Holds | Numeric Control | Defines the maximum time a sustained step (Hold) can be held.	| Controls the duration of long notes. |
| 22 | Máximo de Holds Simultáneos | Numeric Control |	Determines how many held steps can occur at the same time. | Ideal for removing or increasing complexity in specific areas. |
| 23 - 36 |	(Minas, Fakes, Lifts, Potions, Shields, Rayos, Hiddens) | Probability / Max Numerical Control | Each of these groups controls the probability and maximum number of a specific effect per generated measure. | These are very fine adjustments to create specific patterns (e.g., if you want many traps/minas). |

### F. Final Controls and Output

Final adjustments before running the generation.

| Nº | Field | Type | Description | Key Notes |
| :---: | :---: | :---: | :---: | :---: |
| 42 | Semilla de Generación (Seed)	| Number | If you enter a number, the generator will attempt to create exactly the same steps. | Saving the seed is useful for reproducing perfect or desired results. |
| 43 | Restablecer Parámetros |	Button | Clears all fields and returns them to their default values (by default). | RRecommended if starting from scratch. |
| 18, 19 | Banner Graphic / Video de Fondo | Multimedia Selector | Optional elements for the visual presentation of the track. | Do not affect step generation, only the multimedia output. |
| 44 | Procesar y Exportar Dual Pack | Execution Button | Executes all established configurations to generate the final step file (.sm, .ssc). | AMain action upon finalizing configuration. |
| 45 | Monitor de Densidad en Tiempo Real | Console | Provides an approximate and immediate interpretation of the generated rhythmic result as it is being created. | Serves as visual feedback while adjusting parameters. |
