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
| 5, 6 | **Banner Graphic / Background Video** | Multimedia Selector | Optional visual elements for the track presentation. | Do *not* affect step generation; they only affect the multimedia output. |
| 7 | **Rename Files** | Checkbox/Opt. Field | Allows you to force the step file name using the provided title. | Useful for organizing the final pack structure. |
| 8 | **AI Temperature** | Number Control | Determines how "free" or creative the Artificial Intelligence can be when generating steps. | High value = more experimentation; Low value = more conservative and predictable. |
| 9 | **Pack Name** | Text Input | Names the folder where your generated files will be stored. | The generator copies all output files into this specified location. |
| 10 | **Generation Seed (Seed)** | Number Input | If you enter a number, the generator attempts to create exactly the same steps every time. | Saving the seed is useful for reproducing perfect or desired results. |
| 11 | **Templates** | Dropdown Menu | Configuration settings to reference or apply to your generated steps. | Optional. Remember to apply a reset whenever you change templates. |
| 12 | **Hide Parameters** | Dropdown Menu | Organizes the modifier sections into separate, logical areas for easy navigation. | Use this feature to jump directly to the controls you want to modify. |
| 13 | **Reset Parameters** | Button | Clears all fields and reverts them to their default (default) values. | Recommended if starting from scratch. |
| 14 | **Process and Export Dual Pack** | Execution Button | Executes all set configurations to generate the final step file (`.sm`, `.ssc`). | This is the main action button upon completion of configuration. |
| 15 | **Real-Time Density Monitor** | Console Output | Provides an immediate, approximate interpretation of the rhythmic result currently being generated. | Acts as a visual feedback mechanism while you fine-tune parameters. |

#### B. Temporal and Structure Control (Timing Settings)

![Interfaz](./Imagenes/img5.jpg)
![Interfaz](./Imagenes/img6.jpg)

These controls determine the duration of the step map and exactly where the generation should begin.

| # | Field | Type | Description | Key Notes |
| :---: | :---: | :---: | :---: | :---: |
| 16 | **Adjust Limits in Interactive Graph** | Button / Interface | Opens a graphical view to synchronize values for duration, offset, and extension. | Use this if you are new to these technical concepts. |
| 17 | **Calculate and Synchronize Values** | Button | Applies the visualized values from the graph (point 16) to their respective input fields. | Crucial step if using the graphical interface. |
| 18 | **Maximum Duration** | Text Input | Defines the absolute maximum length for the generated step file. | Useful for trimming or limiting the overall extension of the map. |
| 19 | **Start Offset** | Selector/Buttons | Allows manual definition of the exact point where step generation should begin, separate from the total duration. Adjustable with incremental buttons (-0.04s and +0.04s). | Useful if the track starts quietly or without a clear beat. |
| 20 | **Automatically Detect Offset** | Checkbox | If unchecked, you must specify an offset manually (points 15 and 16). | Recommended to disable this if you know the precise starting point of the rhythm. |
| 21 | **Aesthetic Final Extension** | Number Control | Allows extending the audio file *without* generating additional rhythmic steps. | Ideal for creating a smooth "fade out" effect at the end of the track. |

#### C. Master Rhythm and Speed Analysis (BPM & Flow Configuration)

![Interfaz](./Imagenes/img7.jpg)
![Interfaz](./Imagenes/img8.jpg)
![Interfaz](./Imagenes/img9.jpg)

This is the most advanced section, controlling the pulse and responsiveness of the generator based on the audio spectrum (RMS). Understanding **Root Mean Square (RMS)** here means understanding how the software detects changes in volume/energy to match rhythm shifts.

| # | Field | Type | Description | Key Notes |
| :---: | :---: | :---: | :---: | :---: |
| 22 | **BPM Configuration** | Number Control | Sets the desired beats per minute (BPM). Adjustable with incremental buttons (1 BPM at a time). | Defines the basic rhythm foundation of the song. |
| 23 | **Auto** | Checkbox | Enables automatic calculation of the BPM based on audio analysis. | Disable if you intend to apply a manual, fixed BPM. |
| 24 | **Double BPM** | Checkbox | If the calculated or manual BPM is unsatisfactory, this option allows you to double it before step generation. | Advanced rhythmic adjustment. |
| 25 | **Apply Dynamic BPM** | Checkbox | Enables an audio spectrum-based analysis (using RMS) to apply natural changes in BPM throughout the song. | New advanced feature. Adjusts rhythm based on energy peaks. |
| 26 | **Minimum BPM** | Text Input | Configures the minimum allowed BPM for dynamic shifts (point 25). | Advanced, optional setting. If not configured, the software defaults to (Current BPM - 30). |
| 27 | **Maximum BPM** | Text Input | Configures the maximum allowed BPM for dynamic shifts (point 25). | Advanced, optional setting. If not configured, the software defaults to (Current BPM + 30). |
| 28 | **Min RMS Sensitivity BPM** | Number Control | Sets the minimum required audio energy (RMS) level needed for a section to be considered rhythmically notable by the AI. | Specialized for BPM adjustment; impacts quiet sections. |
| 29 | **Max RMS Sensitivity BPM** | Number Control | Sets the maximum peak impact energy (RMS) that defines the strongest rhythmic peak of the song. | Specialized for BPM adjustment; impacts intense peaks. |
| 30 | **Tidal Damper BPM** | Number Control | Sets the approximation value when changing BPMs dynamically. | Higher values result in more abrupt and precise BPM shifts but risk causing "swerving" or audio distortion. |
| 31 | **Adapt Visual Speed** | Checkbox | Enables an audio spectrum-based analysis (using RMS) to apply speed changes (not just tempo). | New experimental feature. Requires manual tuning for optimal results. |
| 32 | **Extended Duration by Loss** | Number Control | Adjusts the map duration when applying speed/tempo changes dynamically. | Advanced: The user must verify this value to ensure the expected song length is covered. |
| 33 | **Min RMS Sensitivity Scroll** | Number Control | Sets the minimum required audio energy (RMS) needed for a section to be considered rhythmically notable by the AI for scrolling speed adjustments. | Specialized for controlling scroll speed in quiet sections. |
| 34 | **Max RMS Sensitivity Scroll** | Number Control | Sets the maximum peak impact energy (RMS) that defines the strongest rhythmic peak of the song for scroll adjustments. | Specialized for controlling scroll speed during intense peaks. |
| 35 | **Speed at Minimums** | Number Control | Sets the minimum velocity parameter applied during the lowest RMS audio moments. | Advanced: Affects how slow or fast the steps are in quiet parts. |
| 36 | **Speed at Maximums** | Number Control | Sets the maximum velocity parameter applied during the highest RMS audio moments. | Advanced: Controls the speed of steps during powerful beats. |
| 37 | **Transition Duration** | Number Control | Determines how long the tool takes to readjust speed when using "Adapt Visual Speed" (point 31). | Controls the smoothness of tempo changes; higher values = smoother, but slower change. |
| 38 | **Anti-Nausea Filter** | Number Control | Sets the threshold for applying a speed change. | Advanced: Higher values ensure stable, less drastic speed shifts; lower values may cause noticeable, jarring cuts or nausea in gameplay. |

#### D. Special Elements and Complexity (Effects, Traps, Mines)

![Interfaz](./Imagenes/img10.jpg)
![Interfaz](./Imagenes/img11.jpg)

These controls allow adding specific elements typical of rhythm game genres to increase complexity and variation.

| # | Field | Type | Description | Key Notes |
| :---: | :---: | :---: | :---: | :---: |
| 39 - 53 | (Minas, Fakes, Lifts, Potions, Shields, Rays, Hiddens) | Probability / Max Num. Control | Each of these groups controls the probability and maximum number of a specific effect per measure generated. | These are extremely fine-tuning adjustments for creating targeted patterns (e.g., if you want many traps/mines). |
| 54 | **Enhance Effects and Traps** | Checkbox | Activating this applies advanced special effects to the steps generated by the AI. | Advanced usage; mandatory for activating complex trap mechanisms. |
| 55 | **Min RMS Sensitivity Traps** | Number Control | Sets the minimum required audio energy (RMS) needed for a section to be considered rhythmically notable by the AI when generating traps. | Specialized for traps in quiet sections. |
| 56 | **Max RMS Sensitivity Traps** | Number Control | Sets the maximum peak impact energy (RMS) that defines the strongest rhythmic peak of the song when generating traps. | Specialized for traps during intense peaks. |

#### E. Difficulty and Note Density (Spectral Filters & Complexity)

![Interfaz](./Imagenes/img12.jpg)
![Interfaz](./Imagenes/img13.jpg)

These controls govern the quality and severity of the generated rhythm pattern, focusing on individual notes and simulated movement.

| # | Field | Type | Description | Key Notes |
| :---: | :---: | :---: | :---: | :---: |
| 57 | **Maximum Hold Duration** | Number Control | Defines the maximum time a sustained step (Hold) can last. | Controls the length of long-held notes. |
| 58 | **Max Simultaneous Holds** | Number Control | Determines how many held steps can occur at the same time. | Useful for controlling complexity in specific areas (e.g., removing or adding density). |
| 59 | **Apply Rhythmic Post-Processing** | Checkbox | Improves the raw output of the AI using additional effects. Disabling this results in a "raw" feeling on holds. | Recommended to activate for higher quality output. |
| 60 | **Generate Jump Sections** | Checkbox | Enables the option of adding more dedicated jump sections into the map. | Use if you are unsatisfied with the current jump frequency. *Reminder: Lowering hold duration is recommended.* |
| 61 | **Min RMS Sensitivity Jumps** | Number Control | Sets the minimum required audio energy (RMS) needed for a section to be considered rhythmically notable by the AI when generating jumps. | Specialized for jumps in quiet sections. |
| 62 | **Max RMS Sensitivity Jumps** | Number Control | Sets the maximum peak impact energy (RMS) that defines the strongest rhythmic peak of the song when generating jumps. | Specialized for jumps during intense peaks. |
| 63 | **Pack Ceiling Difficulty** | Number Control | Defines the general desired difficulty level for the entire step pack. This is used in calculating the final difficulty rating (point 64). | Sets the artistic intention for the final result. |
| 64 | **Recalculate Difficulty Dynamically** | Checkbox/Opt. | Allows recalculating the difficulty based on user's manual choices (e.g., BPM, Scroll Min/Max settings). | If disabled, a fixed default configuration is used for difficulty calculation. |
| 65 | **Min RMS Sensitivity (Note Density)** | Number Control | Sets the minimum required audio energy (RMS) needed for a section to be considered rhythmically notable by the AI when calculating note density. | Specialized for controlling notes in quiet sections. |
| 66 | **Max RMS Sensitivity (Note Density)** | Number Control | Sets the maximum peak impact energy (RMS) that defines the strongest rhythmic peak of the song when calculating note density. | Specialized for controlling notes during intense peaks. |
| 67 | **Lines per Measure** | Buttons | Increases the number of divisions used when drawing notes in a measure. | Advanced: Impacts both complexity and the final difficulty level. |
| 68 | **MIN Notes per Measure** | Buttons | The minimum number of notes considered based on low RMS energy levels. | Highly advanced, high impact on difficulty. Experimental feature. |
| 69 | **MAX Notes per Measure** | Buttons | The maximum number of notes considered based on high RMS energy levels. | Highly advanced, high impact on difficulty. Experimental feature. |
