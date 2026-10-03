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

The interface has been logically divided into sections—from primary inputs down to highly detailed adjustments for rhythm and AI performance.

#### A. Basic Controls and Metadata

![Interfaz](./Imagenes/img1.jpg)
![Interfaz](./Imagenes/img2.jpg)

These fields define what will be generated in terms of file names, artist, and basic settings.

| # | Field | Type | Description | Key Notes |
| :---: | :---: | :---: | :---: | :---: |
| 1 | **Select Song** | Audio Selector | Loads the audio track that will serve as the base for step generation. | Mandatory input. |
| 2 | **Select AI Checkpoint** | File Selector | Loads the model (checkpoint) generated during Phase 1. | Mandatory in this version of the software. |
| 3 | **Song Title** | Text Input | The name assigned to your musical piece. Used for file renaming. | Suggestion: Keep it concise. |
| 4 | **Artist Name** | Text Input | Names the song's artist. | Optional. |
| 5, 6, 7, 8 | **Banner Graphic / Background Video / Background / CdTitle Graphic** | Multimedia Selector | Optional visual elements for the track presentation. | Do *not* affect step generation; they only affect the multimedia output. |
| 9 | **Rename Files** | Checkbox/Opt. Field | Allows you to force the step file name using the provided title. | Useful for organizing the final pack structure. |
| 10 | **AI Temperature** | Number Control | Determines how "free" or creative the Artificial Intelligence can be when generating steps. | High value = more experimentation; Low value = more conservative and predictable. |
| 11 | **Minimum Volume for Silences** | Number Control| Determines how apply notes in silence sections. | Useful for omiting this sections. |
| 12 | **Active Silence Function** | Number Control | Activates the function to read silences. | Optional. Advanced. May obstruct step generation, but it helps skip silent parts of the song. |
| 13 | **Pack Name** | Text Input | Names the folder where your generated files will be stored. | The generator copies all output files into this specified location. |
| 14 | **Generation Seed (Seed)** | Number Input | If you enter a number, the generator attempts to create exactly the same steps every time. | Saving the seed is useful for reproducing perfect or desired results. |
| 15, 16, 17 | **Presets Template / Save Present / Delete Present** | Dropdown Menu | Configuration settings to reference or apply to your generated steps. | Optional. Remember to apply a reset whenever you change templates. |
| 18 | **Hide Parameters** | Dropdown Menu | Organizes the modifier sections into separate, logical areas for easy navigation. | Use this feature to jump directly to the controls you want to modify. |
| 19 | **Reset Parameters** | Button | Clears all fields and reverts them to their default (default) values. | Recommended if starting from scratch. |
| 20 | **Process and Export Dual Pack** | Execution Button | Executes all set configurations to generate the final step file (`.sm`, `.ssc`). | This is the main action button upon completion of configuration. |
| 21 | **Processing Files per Batch** | Execution Button | Executes all stability settings to generate the final step file (.sm .scc) for all folders containing at least one audio file (background, banner, video, cdtitle can be added). | Useful if you already know how the engine works, use the sample option, and have more than one audio file for which you want to generate a step file. Since it involves multiple files, it takes longer to process. |
| 22 | **Generation Cancel** | Button | Cancel operations in process. | Useful for stopping generation. |
| 23 | **Real-Time Density Monitor** | Console Output | Provides an immediate, approximate interpretation of the rhythmic result currently being generated. | Acts as a visual feedback mechanism while you fine-tune parameters. |

#### B. Temporal and Structure Control (Timing Settings)

![Interfaz](./Imagenes/img5.jpg)
![Interfaz](./Imagenes/img6.jpg)

These controls determine the duration of the step map and exactly where the generation should begin.

| # | Field | Type | Description | Key Notes |
| :---: | :---: | :---: | :---: | :---: |
| 24 | **Adjust Limits in Interactive Graph** | Button / Interface | Opens a graphical view to synchronize values for duration, offset, and extension. | Use this if you are new to these technical concepts. |
| 25 | **Calculate and Synchronize Values** | Button | Applies the visualized values from the graph (point 16) to their respective input fields. | Crucial step if using the graphical interface. |
| 26 | **Maximum Duration** | Text Input | Defines the absolute maximum length for the generated step file. | Useful for trimming or limiting the overall extension of the map. |
| 27 | **Starting Offset** | Selector/Buttons | Allows manual definition of the exact point where step generation should begin, separate from the total duration. Adjustable with incremental buttons (-0.04s and +0.04s). | Useful if the track starts quietly or without a clear beat. |
| 28 | **Auto-Detect Offset** | Checkbox | If unchecked, you must specify an offset manually (points 15 and 16). | Recommended to disable this if you know the precise starting point of the rhythm. |
| 29 | **Added Time Percentage** | Numeric Control | Sets a readjustment in map duration when applying speed modifiers (BPM, Scroll Speeds, Stops). This part is inferred using the song's BPM and AI generation. Advanced. Requires the user to verify this value so that it covers the expected duration of the song. Useful if you use Dynamic BPM and Velocity or Stops modes, as well as the sampling function. Reminder: the silence function may interfere with this addition. |
| 30 | **Aesthetic Final Extension** | Number Control | Allows extending the audio file *without* generating additional rhythmic steps. | Ideal for creating a smooth "fade out" effect at the end of the track. |
| 31 | **Extension by Holders? Mines by Default** | Checkbox | Adds the possibility for the aesthetic extension to be used as a closure for holders. By default, it extends with mines. | Optional. Advanced. Requires the user to verify the result to coordinate that this section fits correctly at the exact moment they want it to appear. |

#### C. Master Rhythm and Speed Analysis (BPM & Flow Configuration)

![Interfaz](./Imagenes/img7.jpg)
![Interfaz](./Imagenes/img8.jpg)
![Interfaz](./Imagenes/img9.jpg)

This is the most advanced section, controlling the pulse and responsiveness of the generator based on the audio spectrum (RMS). Understanding **Root Mean Square (RMS)** here means understanding how the software detects changes in volume/energy to match rhythm shifts.

| # | Field | Type | Description | Key Notes |
| :---: | :---: | :---: | :---: | :---: |
| 32 | **BPM Configuration** | Number Control | Sets the desired beats per minute (BPM). Adjustable with incremental buttons (1 BPM at a time). | Defines the basic rhythm foundation of the song. |
| 33 | **Auto** | Checkbox | Enables automatic calculation of the BPM based on audio analysis. | Disable if you intend to apply a manual, fixed BPM. |
| 34 | **Double BPM** | Checkbox | If the calculated or manual BPM is unsatisfactory, this option allows you to double it before step generation. | Advanced rhythmic adjustment. |
| 35 | **Apply x2 starting at a BPM lower than** | Text Input | Used to check that the BPM is lower than this value for duplication. | Optional. By default, if you leave this field blank, it compares against a BPM of 300. Useful for batch generation mode. |
| 36 | **Apply Dynamic BPM** | Checkbox | Enables an audio spectrum-based analysis (using RMS) to apply natural changes in BPM throughout the song. | New advanced feature. Adjusts rhythm based on energy peaks. |
| 37 | **Minimum BPM** | Text Input | Configures the minimum allowed BPM for dynamic shifts (point 25). | Advanced, optional setting. If not configured, the software defaults to (Current BPM - 30). |
| 38 | **Maximum BPM** | Text Input | Configures the maximum allowed BPM for dynamic shifts (point 25). | Advanced, optional setting. If not configured, the software defaults to (Current BPM + 30). |
| 39 | **Minimum BPM RMS Sensitivity** | Number Control | Sets the minimum required audio energy (RMS) level needed for a section to be considered rhythmically notable by the AI. | Specialized for BPM adjustment; impacts quiet sections. |
| 40 | **Maximum BPM RMS Sensitivity** | Number Control | Sets the maximum peak impact energy (RMS) that defines the strongest rhythmic peak of the song. | Specialized for BPM adjustment; impacts intense peaks. |
| 41 | **MIN BPM per BPM Change** | Numeric Control | Sets a minimum change of BPM in the reading of the audio signal to write the change to the final file. Usage for advanced users. The lower it is, the more reactive it is, but it is more likely that your file will break when loaded into the game. If this happens, delete that file (.scc) and generate a new one with different configurations. Files (.sm) should not have problems. |
| 42 | **BPM Tide Damping** | Number Control | Sets the approximation value for BPM changes. The higher it is, the more sudden and exact the BPM change, but it can cause dizziness or distortions. This field brings BPM changes closer to the values in the Minimum and Maximum BPM fields. |
| 43 | **Adapt Visual Speed** | Checkbox | Enables a spectrum based on RMS (reading of high and low audio peaks) to apply speed changes. New option. Experimental function. Advanced rhythm adjustment. Activating this option requires manual adjustment by the user. Does not apply to (.sm) files. |
| 44 | **Minimum Scroll RMS Sensitivity** | Number Control | Sets the minimum required audio energy (RMS) needed for a section to be considered rhythmically notable by the AI for scrolling speed adjustments. | Specialized for controlling scroll speed in quiet sections. |
| 45 | **Maximum Scroll RMS Sensitivity** | Number Control | Sets the maximum peak impact energy (RMS) that defines the strongest rhythmic peak of the song for scroll adjustments. | Specialized for controlling scroll speed during intense peaks. |
| 46 | **Speed at Minimums** | Number Control | Sets the minimum velocity parameter applied during the lowest RMS audio moments. | Advanced: Affects how slow or fast the steps are in quiet parts. |
| 47 | **Speed at Maximums** | Number Control | Sets the maximum velocity parameter applied during the highest RMS audio moments. | Advanced: Controls the speed of steps during powerful beats. |
| 48 | **Transition Duration** | Number Control | Determines how long the tool takes to readjust speed when using "Adapt Visual Speed" (point 31). | Controls the smoothness of tempo changes; higher values = smoother, but slower change. |
| 49 | **Anti-Dizziness Filter** | Numeric Control | Sets the threshold for applying a speed change. Advanced. The higher the value, the more stable the speed change; the lower it is, the more abrupt cuts and dizziness can occur. Also, similar to BPM, it becomes more reactive and may break the game. If this happens, delete the file (.scc) and create a new one with different configurations. Sets the approximation value for applying Minimum Speed or Maximum Speed. |
| 50 | **Enable Dynamic Stop Usage**| Checkbox | Enables the stop function in the step file. Does not apply to the (.sm) file. Enabling it in case of wanting to use this option; depending on the song duration, the final file may corrupt the Stepmania engine or its derivatives. If this happens, delete the file (.scc) and use other configurations. |
| 51 | **Stop Activation Threshold** | Numeric Control | Determines the minimum reading in the audio signal to apply a stop. The lower it is, the more reactive. |
| 52 | **Stop Duration** | Numeric Control | Determines the duration of a Stop. Recommended to use a low value. |

#### D. Special Elements and Complexity (Effects, Traps, Mines)

![Interfaz](./Imagenes/img10.jpg)
![Interfaz](./Imagenes/img11.jpg)

These controls allow adding specific elements typical of rhythm game genres to increase complexity and variation.

| # | Field | Type | Description | Key Notes |
| :---: | :---: | :---: | :---: | :---: |
| 53 - 66 | (Minas, Fakes, Lifts, Potions, Shields, Rays, Hiddens) | Probability / Max Num. Control | Each of these groups controls the probability and maximum number of a specific effect per measure generated. | These are extremely fine-tuning adjustments for creating targeted patterns (e.g., if you want many traps/mines). |
| 67 | **Enhance Effects and Traps** | Checkbox | Activating this applies advanced special effects to the steps generated by the AI. | Advanced usage; mandatory for activating complex trap mechanisms. |
| 68 | **Minimum Traps RMS Sensitivity** | Number Control | Sets the minimum required audio energy (RMS) needed for a section to be considered rhythmically notable by the AI when generating traps. | Specialized for traps in quiet sections. |
| 69 | **Maximum Traps RMS Sensitivity** | Number Control | Sets the maximum peak impact energy (RMS) that defines the strongest rhythmic peak of the song when generating traps. | Specialized for traps during intense peaks. |

#### E. Difficulty and Note Density (Spectral Filters & Complexity)

![Interfaz](./Imagenes/img14.jpg)
![Interfaz](./Imagenes/img12.png)
![Interfaz](./Imagenes/img13.jpg)

These controls govern the quality and severity of the generated rhythm pattern, focusing on individual notes and simulated movement.

| # | Field | Type | Description | Key Notes |
| :---: | :---: | :---: | :---: | :---: |
| 70 | **Enable Multi-Layer Sample Generation** | Checkbox | Allows the creation of multiple files by taking a percentage of the total steps. | Usalo para afinar el mapa si no quieres usar el editor de Stepmania o acelerar el proceso para ajustar las configuraciones que estas buscando. Útil entre más muestras uses y a un porcentaje menor de muestreo (Ej: 90%). |
| 71 | **Minimum Initial Sampling** | Number Control | The percentage of total steps from which the sampling process begins. | It is recommended to set this value between 90% and 95%, which covers the final portion of the song's duration. |
| 72 | **Total Intermediate Samples** | Number Control | Defines the number of samples to be generated, starting from the minimum percentage up to 100% of the total duration. | Set this value between 5 and 10 for greater precision. |
| 73 | **Enable Multi-threaded Execution** | Checkbox | Activa la función de multi núcleo, usando todos los recursos de CPU. | Recomendado activarlo en la función de muestreo o por lotes. |
| 74 | **Maximum Hold Duration** | Number Control | Defines the maximum time a sustained step (Hold) can last. | Controls the length of long-held notes. |
| 75 | **Max Simultaneous Holds** | Number Control | Determines how many held steps can occur at the same time. | Useful for controlling complexity in specific areas (e.g., removing or adding density). |
| 76 | **Apply Rhythmic Post-Processing to Holders** | Checkbox | Cleans and corrects the AI's base holds to the established configurations. Disabling it produces a "raw" result in the holds. | Recommended to activate for better quality. |
| 77 | **Generate Jump Sections** | Checkbox | Enables the option of adding more dedicated jump sections into the map. | Use if you are unsatisfied with the current jump frequency. *Reminder: Lowering hold duration is recommended.* |
| 78 | **Minimum Jumps RMS Sensitivity** | Number Control | Sets the minimum required audio energy (RMS) needed for a section to be considered rhythmically notable by the AI when generating jumps. | Specialized for jumps in quiet sections. |
| 79 | **Maximum Jumps RMS Sensitivity** | Number Control | Sets the maximum peak impact energy (RMS) that defines the strongest rhythmic peak of the song when generating jumps. | Specialized for jumps during intense peaks. |
| 80 | **Pack Ceiling Difficulty** | Number Control | Defines the general desired difficulty level for the entire step pack. This is used in calculating the final difficulty rating (point 64). | Sets the artistic intention for the final result. |
| 81 | **Recalculate Difficulty Dynamically** | Checkbox/Opt. | Allows recalculating the difficulty based on user's manual choices (e.g., BPM, Scroll Min/Max settings). | If disabled, a fixed default configuration is used for difficulty calculation. |
| 82 | **Minimum RMS Sensitivity (Note Density)** | Number Control | Sets the minimum required audio energy (RMS) needed for a section to be considered rhythmically notable by the AI when calculating note density. | Specialized for controlling notes in quiet sections. |
| 83 | **Maximum RMS Sensitivity  (Note Density)** | Number Control | Sets the maximum peak impact energy (RMS) that defines the strongest rhythmic peak of the song when calculating note density. | Specialized for controlling notes during intense peaks. |
| 84 | **Lines per Measure** | Buttons | Increases the number of divisions used when drawing notes in a measure. | Advanced: Impacts both complexity and the final difficulty level. |
| 85 | **MIN Notes per Measure** | Buttons | The minimum number of notes considered based on low RMS energy levels. | Highly advanced, high impact on difficulty. Experimental feature. |
| 86 | **MAX Notes per Measure** | Buttons | The maximum number of notes considered based on high RMS energy levels. | Highly advanced, high impact on difficulty. Experimental feature. |

#### F. Batch Processing Section (Batch Mode Filters)

| # | Field | Type | Description | Key Notes |
| :---: | :---: | :---: | :---: | :---: |
| 87 | **Keyword to Locate Banner** | Text | Assign a word that must be in your banner file name to identify and use it as a banner. It is recommended to configure your files with keywords for easy location. By default, if you do not put anything in this field, during batch processing, the first image found will be taken as the banner in this order of priority (Background, Banner, CdTitle). |
| 88 | **Keyword to Locate Video** | Text | Assign a word that must be in your video file name to identify and use it as a video. It is recommended to configure your files with keywords for easy location. By default, if you do not put anything in this field, during batch processing, the first video found will be taken. |
| 89 | **Keyword to Locate Background** | Text | Assign a word that must be in your background file name to identify and use it as a background. It is recommended to configure your files with keywords for easy location. By default, if you do not put anything in this field, during batch processing, the first image found will be taken as the background in this order of priority (Background, Banner, CdTitle). |
| 90 | **Keyword to Locate CD Title** | Text | Assign a word that must be in your cdtitle file name to identify and use it as a cdtitle. It is recommended to configure your files with keywords for easy location. By default, if you do not put anything in this field, during batch processing, the first image found will be taken as the cdtitle in this order of priority (Background, Banner, CdTitle). |
