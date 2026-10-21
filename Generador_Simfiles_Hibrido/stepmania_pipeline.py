# -*- coding: utf-8 -*-
import os
import random
import librosa
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# =====================================================================
# 1. TOKENIZADOR DE SECUENCIAS DE PASOS
# =====================================================================
class StepTokenizer:
    def __init__(self):
        caracteres_validos = ['0', '1', '2', '3', '4']
        todas_combinaciones = list(itertools.product(caracteres_validos, repeat=4))
        self.token_to_id = {}
        self.id_to_token = {}
        self.PAD_TOKEN = 0
        self.BOS_TOKEN = 1
        self.EOS_TOKEN = 2
        for idx, comb in enumerate(todas_combinaciones, start=3):
            token_str = "".join(comb)
            self.token_to_id[token_str] = idx
            self.id_to_token[idx] = token_str

    def encode_linea(self, lista_cuatro_chars):
        token_str = "".join(lista_cuatro_chars)
        return self.token_to_id.get(token_str, self.PAD_TOKEN)

    def decode_id(self, token_id):
        if isinstance(token_id, torch.Tensor):
            token_id = token_id.item()
        if token_id in self.id_to_token:
            return list(self.id_to_token[token_id])
        return ['0', '0', '0', '0']

# =====================================================================
# 2. MÓDULO DE AUMENTO DE DATOS EN AUDIO (DATA AUGMENTATION)
# =====================================================================
class AudioAugmentationPipeline:
    @staticmethod
    def agregar_ruido_gaussiano(y, min_snr=15, max_snr=30):
        rms_y = np.sqrt(np.mean(y**2)) if len(y) > 0 else 1.0
        snr_elegido = random.uniform(min_snr, max_snr)
        rms_ruido = rms_y / (10 ** (snr_elegido / 20))
        ruido = np.random.normal(0, rms_ruido, y.shape)
        return y + ruido

    @staticmethod
    def alterar_tono(y, sr, min_steps=-2, max_steps=2):
        steps = random.randint(min_steps, max_steps)
        if steps == 0: 
            return y
        return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=steps)

    @staticmethod
    def invertir_polaridad(y):
        return y * -1.0

    @classmethod
    def aplicar_aumento_aleatorio(cls, y, sr, probabilidad=0.5):
        if random.random() > probabilidad: 
            return y
        y_aug = y.copy()
        transformaciones = [cls.agregar_ruido_gaussiano, cls.alterar_tono, cls.invertir_polaridad]
        random.shuffle(transformaciones)
        for trans in transformaciones:
            if random.random() > 0.5:
                if trans == cls.alterar_tono: 
                    y_aug = trans(y_aug, sr)
                else: 
                    y_aug = trans(y_aug)
        return y_aug

# =====================================================================
# PARSER HÍBRIDO AVANZADO (PROXIMIDAD TEMPORAL DE GIMMICKS)
# =====================================================================
def parsear_simfile_generico(archivo_path, audio_folder_path, n_mels=64, aplicar_augmentation=False):
    """
    Lee archivos (.sm/.ssc/.sma) procesando los compases de forma aislada e inyectando
    el estado síncrono local de BPM, Speeds, Stops y Delays en el vector de audio.
    """
    try:
        with open(archivo_path, 'r', encoding='utf-8', errors='ignore') as f:
            contenido = f.read()
    except Exception as e:
        print(f" ❌ Error leyendo archivo: {archivo_path}: {e}")
        return None

    bloques = [b.strip() for b in contenido.split(';') if b.strip()]
    metadata_global = {}
    charts_raw = []
    es_ssc = archivo_path.lower().endswith('.ssc')
    bloque_chart_actual = []

    for b in bloques:
        if b.startswith('#NOTES:'):
            if es_ssc: bloque_chart_actual.append(b)
            else: charts_raw.append([b])
        elif b.startswith('#NOTEDATA:'):
            if bloque_chart_actual: charts_raw.append(bloque_chart_actual)
            bloque_chart_actual = [b]
        elif es_ssc and bloque_chart_actual:
            bloque_chart_actual.append(b)
        elif ':' in b:
            partes = b.split(':', 1)
            if len(partes) == 2: 
                metadata_global[partes[0].strip()] = partes[1].strip()

    if es_ssc and bloque_chart_actual:
        charts_raw.append(bloque_chart_actual)

    audio_name = metadata_global.get('#MUSIC', '').replace(';', '').strip()
    if not audio_name: return None
    audio_path = os.path.join(audio_folder_path, audio_name)
    if not os.path.exists(audio_path): return None

    # --- EXTRACCIÓN Y MAPEO DISCRETO DE GIMMICKS ---
    def parsear_tabla_gimmick(tag_name):
        tabla = []
        if tag_name in metadata_global:
            raw_data = metadata_global[tag_name].replace(f'{tag_name}:', '').replace(';', '').strip()
            for par in raw_data.split(','):
                if '=' in par:
                    try:
                        partes = par.split('=')
                        # Estructura básica: Beat=Valor (o Beat=Valor=Duración... en SSC)
                        tabla.append((float(partes[0].strip()), float(partes[1].strip())))
                    except: pass
        return tabla

    mapa_bpms = parsear_tabla_gimmick('#BPMS')
    mapa_speeds = parsear_tabla_gimmick('#SPEEDS')
    mapa_stops = parsear_tabla_gimmick('#STOPS')
    mapa_delays = parsear_tabla_gimmick('#DELAYS')

    # Fallbacks de seguridad si las tablas vienen vacías
    if not mapa_bpms: mapa_bpms.append((0.0, 120.0))
    if not mapa_speeds: mapa_speeds.append((0.0, 1.0))

    # Buscadores de estado síncrono por proximidad rítmica (Beat actual)
    def buscar_valor_gimmick(tabla, beat_actual, valor_defecto=0.0):
        valor_local = valor_defecto
        for b_beat, b_val in tabla:
            if beat_actual >= b_beat: valor_local = b_val
            else: break
        return valor_local

    def verificar_congelamiento_activo(tabla, beat_actual):
        # En StepMania, Stops/Delays se listan como Beat=SegundosDeCongelamiento
        for b_beat, b_segundos in tabla:
            # Si el beat coincide exactamente o estamos en el rango inmediato del impacto del gimmick
            if abs(beat_actual - b_beat) < 0.001 and b_segundos > 0:
                return b_segundos
        return 0.0

    # Carga y análisis DSP de la canción
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        lista_rms = librosa.feature.rms(y=y).flatten()
        lista_centroide = librosa.feature.spectral_centroid(y=y, sr=sr).flatten()
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        rms_medio = np.mean(lista_rms) if len(lista_rms) > 0 else 1.0
        centroide_medio = np.mean(lista_centroide) if len(lista_centroide) > 0 else 1.0
    except Exception as e:
        print(f" ❌ Error en análisis de audio ({audio_name}): {e}")
        return None

    dataset_de_este_archivo = []

    for sub_bloques in charts_raw:
        cuerpo_notas = ""
        dificultad, nivel = "Challenge", "10"
        if es_ssc:
            for sub_b in sub_bloques:
                if sub_b.startswith('#DIFFICULTY:'): dificultad = sub_b.split(':', 1)[1].strip()
                elif sub_b.startswith('#METER:'): nivel = sub_b.split(':', 1)[1].strip()
                elif sub_b.startswith('#NOTES:'): cuerpo_notas = sub_b.split(':', 1)[1].strip()
        else:
            partes_sm = sub_bloques[0].split(':')
            if len(partes_sm) >= 7:
                dificultad = partes_sm[3].strip()
                nivel = partes_sm[4].strip()
                cuerpo_notas = partes_sm[6].strip()

        if not cuerpo_notas: continue

        compases_raw = [c.strip() for c in cuerpo_notas.split(',')]
        secuencia_datos = []
        
        beat_acumulado = 0.0
        segundo_acumulado = 0.0
        ultima_col_simple = -1

        for idx_compas, compas in enumerate(compases_raw):
            lineas_compas = [l.strip() for l in compas.split('\n') if l.strip() and not l.strip().startswith('//')]
            lineas_en_este_compas = len(lineas_compas)
            
            if lineas_en_este_compas == 0:
                beat_acumulado += 4.0
                bpm_local = buscar_valor_gimmick(mapa_bpms, beat_acumulado, 120.0)
                segundo_acumulado += (4.0 * 60.0) / bpm_local
                continue

            beats_por_linea = 4.0 / lineas_en_este_compas

            for idx_linea, linea in enumerate(lineas_compas):
                # 📡 LECTURA DINÁMICA SÍNCRONA DE TELEMETRÍA DE GIMMICKS
                bpm_local = buscar_valor_gimmick(mapa_bpms, beat_acumulado, 120.0)
                scroll_local = buscar_valor_gimmick(mapa_speeds, beat_acumulado, 1.0)
                stop_segundos = verificar_congelamiento_activo(mapa_stops, beat_acumulado)
                delay_segundos = verificar_congelamiento_activo(mapa_delays, beat_acumulado)

                duracion_linea_segundos = (beats_por_linea * 60.0) / bpm_local
                
                idx_audio = min(int((segundo_acumulado / (len(lista_rms) * (512/22050))) * len(lista_rms)), len(lista_rms) - 1)
                idx_audio = max(0, idx_audio)

                flechas_token = [char for char in linea[:4]]
                if len(flechas_token) < 4: flechas_token = ["0", "0", "0", "0"]
                
                # Filtro biomecánico de limpieza pasivo
                activas = [i for i, c in enumerate(flechas_token) if c in ['1', '2']]
                if len(activas) >= 3:
                    flechas_token = ["0", "0", "0", "0"]
                    col_elegida = random.choice([0, 1, 2, 3])
                    flechas_token[col_elegida] = "1"
                    activas = [col_elegida]

                if len(activas) == 1 and flechas_token[activas[0]] == '1':
                    if activas[0] == ultima_col_simple:
                        columnas_libres = [c for c in range(4) if c != ultima_col_simple]
                        flechas_token = ["0", "0", "0", "0"]
                        col_nueva = random.choice(columnas_libres)
                        flechas_token[col_nueva] = "1"
                        ultima_col_simple = col_nueva
                    else:
                        ultima_col_simple = activas[0]

                # Consolidación del diccionario intermedio de empaquetado
                secuencia_datos.append({
                    "energia_ratio": round(lista_rms[idx_audio] / rms_medio, 3),
                    "brillo_ratio": round(lista_centroide[idx_audio] / centroide_medio, 3),
                    "vector_mel": mel_db[:, idx_audio].tolist(),
                    "patron_humano": flechas_token,
                    # --- NUEVOS CAMPOS DEL MOTOR MULTI-CANAL ---
                    "contexto_lineas": lineas_en_este_compas,
                    "g_bpm": bpm_local,
                    "g_scroll": scroll_local,
                    "g_stop_delay": max(stop_segundos, delay_segundos) # Combina congelamientos activos
                })

                beat_acumulado += beats_por_linea
                # Si ocurrió un Stop/Delay físico en esta casilla, el tiempo de audio se desplaza hacia adelante
                segundo_acumulado += (duracion_linea_segundos + stop_segundos + delay_segundos)

        dataset_de_este_archivo.append({
            "cancion": metadata_global.get('#TITLE', 'Unknown').replace(';', '').strip(),
            "dificultad": dificultad,
            "nivel": nivel,
            "secuencia": secuencia_datos
        })

    return dataset_de_este_archivo

# =====================================================================
# PYTORCH DATASET CON COMPOSICIÓN EXPANDIDA DE CANALES (INPUT_DIMS = 70)
# =====================================================================
class StepManiaTorchDataset(Dataset):
    def __init__(self, datos_crudos, seq_len=128, n_mels=64, tokenizer=None):
        self.seq_len = seq_len
        self.n_mels = n_mels
        self.tokenizer = tokenizer if tokenizer else StepTokenizer()
        self.features_audio = []
        self.target_tokens = []
        self._procesar_y_segmentar(datos_crudos)

    def _procesar_y_segmentar(self, datos_crudos):
        for item in datos_crudos:
            secuencia = item["secuencia"]
            if len(secuencia) < self.seq_len: continue

            energia = [p["energia_ratio"] for p in secuencia]
            brillo = [p["brillo_ratio"] for p in secuencia]
            # --- NORMALIZACIÓN DE CARACTERÍSTICAS BIOMECÁNICAS/GIMMICKS ---
            lineas_contexto = [p["contexto_lineas"] / 192.0 for p in secuencia]
            telemetria_bpm = [min(300.0, p["g_bpm"]) / 300.0 for p in secuencia]      # Escalado estricto a base 300
            telemetria_scroll = [min(4.0, p["g_scroll"]) / 4.0 for p in secuencia]    # Escalado estricto a base 4x
            telemetria_congelado = [min(2.0, p["g_stop_delay"]) / 2.0 for p in secuencia] # Escalado estricto a base 2s

            espectro_mel = [p["vector_mel"] for p in secuencia]
            tokens = [self.tokenizer.encode_linea(p["patron_humano"]) for p in secuencia]
            
            total_pasos = len(tokens)

            if total_pasos < self.seq_len: 
                continue

            # Segmentación de ventanas móviles
            for i in range(0, total_pasos - self.seq_len + 1, self.seq_len // 2):
                fin = i + self.seq_len
                
                # Stack de variables base expandidas. Dimensiones estructurales: (seq_len, 6)
                base_features = np.stack([
                energia[i:fin],
                brillo[i:fin],
                lineas_contexto[i:fin],
                telemetria_bpm[i:fin],
                telemetria_scroll[i:fin],
                telemetria_congelado[i:fin]
                ], axis=-1)

                mel_block = np.array(espectro_mel[i:fin])
                
                # Ensamblado Final: 6 Canales Gimmick/DSP + 64 Canales Mel = 70 Dimensiones de Entrada Únicas
                self.features_audio.append(np.concatenate([base_features, mel_block], axis=-1))
                self.target_tokens.append(tokens[i:fin])
        
        if self.features_audio:
            self.features_audio = torch.tensor(np.array(self.features_audio), dtype=torch.float32)
            self.target_tokens = torch.tensor(np.array(self.target_tokens), dtype=torch.long)
        else:
            # Fallback seguro
            self.features_audio = torch.empty((0, self.seq_len, ADITIONAL_CHANNELS + self.n_mels), dtype=torch.float32)
            self.target_tokens = torch.empty((0, self.seq_len), dtype=torch.long)

    def __len__(self): 
        return len(self.target_tokens)

    def __getitem__(self, idx): 
        return self.features_audio[idx], self.target_tokens[idx]

# =====================================================================
# 5. ORQUESTADOR CENTRAL: ESCANEO Y VALIDACIÓN CRUZADA (TRAIN/TEST SPLIT)
# =====================================================================
class StepManiaDataOrchestrator:
    def __init__(self, canciones_dir, seq_len=128, n_mels=64, split_ratio=0.85):
        self.canciones_dir = canciones_dir
        self.seq_len = seq_len
        self.n_mels = n_mels
        self.split_ratio = split_ratio
        self.tokenizer = StepTokenizer()

    def construir_datasets_clasificados(self, test_size=0.15, duplicar_train_aug=True, max_workers=4):
        """
        Construye los datasets divididos utilizando un ThreadPoolExecutor 
        para acelerar el procesamiento de canciones en paralelo.
        """
        carpetas_canciones = []
        for raiz, dirnames, archivos in os.walk(self.canciones_dir):
            # Se actualiza la tupla de extensiones permitidas para incluir .sma
            if any(f.lower().endswith(('.sm', '.ssc', '.sma')) for f in archivos):
                # Guardamos una tupla con la ruta y sus subcarpetas
                carpetas_canciones.append((raiz, dirnames))

        if not carpetas_canciones:
            print(" ❌ No se encontraron canciones válidas (.sm, .ssc o .sma) en las subcarpetas.")
            return None, None

        random.seed(42)
        random.shuffle(carpetas_canciones)
        
        limite = int(len(carpetas_canciones) * (1.0 - test_size))
        carpetas_train = carpetas_canciones[:limite]
        carpetas_val = carpetas_canciones[limite:]
        
        print(f" 📂 Carpetas detectadas: {len(carpetas_canciones)} -> Entrenamiento: {len(carpetas_train)} | Validación: {len(carpetas_val)}")

        # --- FUNCIÓN INTERNA OPTIMIZADA CON MULTIHILO Y TQDM ---
        def extraer_datos_de_lista_multithread(lista_carpetas, augmentation, descripcion_pbar="Cargando"):
            datos_acumulados = []
            
            # Definimos la tarea atómica que ejecutará cada hilo por carpeta
            def procesar_carpeta_individual(tupla_carpeta):
                carpeta, subcarpetas = tupla_carpeta
                resultados_locales = []
                
                # Escaneo interno incluyendo archivos .sma
                for archivo in os.listdir(carpeta):
                    if archivo.lower().endswith(('.sm', '.ssc', '.sma')):
                        res = parsear_simfile_generico(
                            os.path.join(carpeta, archivo), 
                            carpeta,
                            self.n_mels, 
                            augmentation
                        )
                        if res:
                            resultados_locales.extend(res)
                return resultados_locales, carpeta, len(subcarpetas)

            # Inicializamos el ejecutor de hilos y la barra de progreso
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Mapeamos todas las carpetas a los hilos concurrentes
                futuros = {executor.submit(procesar_carpeta_individual, item): item for item in lista_carpetas}
                
                with tqdm(total=len(lista_carpetas), desc=descripcion_pbar, unit="canción") as pbar:
                    for futuro in as_completed(futuros):
                        try:
                            resultados_locales, carpeta, num_subdirs = futuro.result()
                            if resultados_locales:
                                datos_acumulados.extend(resultados_locales)
                            
                            # Actualización segura del estado visual en consola de forma dinámica
                            nombre_carpeta_corta = os.path.basename(carpeta)
                            pbar.set_postfix_str(f"Carpeta: {nombre_carpeta_corta} (Subdirs: {num_subdirs})")
                        except Exception as e:
                            print(f"\n ❌ Error procesando un hilo de ejecución: {e}")
                        finally:
                            pbar.update(1) # Avanza un paso en la barra global por cada carpeta completada
                            
            return datos_acumulados

        # Llamadas concurrentes pasando el número máximo de trabajadores (hilos)
        datos_train = extraer_datos_de_lista_multithread(
            carpetas_train, 
            augmentation=duplicar_train_aug, 
            descripcion_pbar=" 🎵 Cargando Entrenamiento (Con Aumento)"
        )
        
        datos_val = extraer_datos_de_lista_multithread(
            carpetas_val, 
            augmentation=False, 
            descripcion_pbar=" 🧹 Cargando Validación (Limpio)"
        )

        return (
            StepManiaTorchDataset(datos_train, self.seq_len, self.n_mels, self.tokenizer),
            StepManiaTorchDataset(datos_val, self.seq_len, self.n_mels, self.tokenizer)
        )

# =====================================================================
# 6. ARQUITECTURA DEL TRANSFORMER MULTICANAL CAUSAL
# =====================================================================
class StepTransformerMulticanalCausal(nn.Module):
    def __init__(self, vocab_size, seq_len, input_dims=70, d_model=128, nhead=4, num_layers=4, dim_feedforward=256):
        super().__init__()
        self.seq_len = seq_len
        self.proyeccion_audio = nn.Linear(input_dims, d_model)
        self.pos_embedding = nn.Embedding(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.clasificador = nn.Linear(d_model, vocab_size)
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
        self.register_buffer('causal_mask', mask)

    def forward(self, src):
        batch_size = src.size(0)
        x = self.proyeccion_audio(src)
        posiciones = torch.arange(0, self.seq_len, device=src.device).unsqueeze(0).repeat(batch_size, 1)
        x = x + self.pos_embedding(posiciones)
        x = self.transformer_encoder(x, mask=self.causal_mask)
        return self.clasificador(x)

# =====================================================================
# 7. VALIDADOR VISUAL EN CONSOLA (ASCII STEP RENDERER)
# =====================================================================
class VisualStepValidator:
    MAPEO_ASCII = {'0': ' . ', '1': ' ◄ ', '2': ' ⇪ ', '3': ' ⇓ ', '4': ' 💣 '}

    @classmethod
    def mapear_caracter_flecha(cls, carril, char):
        iconos = ['←', '↓', '↑', '→']
        if char == '1': return iconos[carril]
        if char == '2': return ' ⇪ '
        if char == '3': return ' ⇓ '
        if char == '4': return ' 💣 '
        return ' . '

    @classmethod
    def imprimir_comparativa(cls, secuencia_real, secuencia_predicha, max_pasos=24):
        print("=================== VALIDADOR VISUAL EN RITMO ===================")
        print(" PASO EN TIEMPO | MAPA HUMANO REAL | PREDICCIÓN DE LA IA")
        print("-" * 65)
        for idx in range(min(max_pasos, len(secuencia_real))):
            real_chars = secuencia_real[idx]
            pred_chars = secuencia_predicha[idx]
            str_real = "".join(cls.mapear_caracter_flecha(i, c) for i, c in enumerate(real_chars))
            str_pred = "".join(cls.mapear_caracter_flecha(i, c) for i, c in enumerate(pred_chars))
            linea_salida = f" Step idx: {idx:03d} | {str_real} | {str_pred}"
            if real_chars == pred_chars and real_chars != ['0', '0', '0', '0']:
                linea_salida += " [Acierto] ⭐"
            print(linea_salida)
        print("=================================================================")

# =====================================================================
# 8. GESTOR DE PUNTOS DE CONTROL (MODEL CHECKPOINTS)
# =====================================================================
class StepManiaCheckpointManager:
    def __init__(self, checkpoint_dir="./checkpoints", model_name="step_transformer_best.pt"):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = os.path.join(checkpoint_dir, model_name)
        self.worse_checkpoint_path = os.path.join(checkpoint_dir, "step_transformer_worse.pt")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.se_uso_checkpoint_existente = False
        self.hubo_mejora_en_esta_sesion = False
        self.ultimo_estado_memoria = None

    def guardar_mejor_modelo(self, modelo, optimizador, epoch, loss_actual, mejor_loss):
        state = {
            'epoch': epoch,
            'model_state_dict': modelo.state_dict(),
            'optimizer_state_dict': optimizador.state_dict(),
            'loss': loss_actual
        }
        self.ultimo_estado_memoria = state

        if loss_actual < mejor_loss:
            print(f" ✨ Nueva menor pérdida validada ({loss_actual:.4f} < {mejor_loss:.4f}). Guardando pesos en best...")
            torch.save(state, self.checkpoint_path)
            self.hubo_mejora_en_esta_sesion = True
            return loss_actual
        return mejor_loss

    def guardar_worse_al_finalizar(self):
        if self.se_uso_checkpoint_existente and not self.hubo_mejora_en_esta_sesion:
            if self.ultimo_estado_memoria is not None:
                print(f"\n🛑 Entrenamiento concluido sin mejoras sobre el checkpoint base.")
                print(f"💾 Guardando el último estado obtenido en: '{self.worse_checkpoint_path}'")
                torch.save(self.ultimo_estado_memoria, self.worse_checkpoint_path)
            else:
                print("\n🛑 Entrenamiento concluido inmediatamente, no se procesaron épocas.")
        elif self.se_uso_checkpoint_existente and self.hubo_mejora_en_esta_sesion:
            print("\n🎉 Entrenamiento concluido con éxito. La IA mejoró, por lo que no se genera archivo 'worse'.")

    def cargar_checkpoint_si_existe(self, modelo, optimizador=None):
        if os.path.exists(self.checkpoint_path):
            print(f" 📦 Cargando pesos desde checkpoint histórico: {self.checkpoint_path}")
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(self.checkpoint_path, map_location=dev)
            modelo.load_state_dict(checkpoint['model_state_dict'])
            
            if optimizador is not None and 'optimizer_state_dict' in checkpoint:
                optimizador.load_state_dict(checkpoint['optimizer_state_dict'])
            
            self.se_uso_checkpoint_existente = True
            return checkpoint.get('epoch', 0), checkpoint.get('loss', float('inf'))
        
        print(" 🆕 Inicializando pesos de forma aleatoria en el Transformer. ")
        self.se_uso_checkpoint_existente = False
        return 0, float('inf')

# =====================================================================
# 9. BUCLE EJECUTABLE PRINCIPAL (ENTRENAMIENTO Y EVALUACIÓN VISUAL)
# =====================================================================
if __name__ == "__main__":
    DIRECTORIO_PACKS = "./StepMania_Songs_Pack/"
    SEQ_LEN = 128
    N_MELS = 64
    ADITIONAL_CHANNELS = 6
    INPUT_DIMS = ADITIONAL_CHANNELS + N_MELS
    VOCAB_SIZE = 628
    
    BATCH_SIZE = 8               
    PASOS_ACUMULACION = 2         
    EPOCHS = 10

    os.makedirs(DIRECTORIO_PACKS, exist_ok=True)
    
    # 1. ASIGNACIÓN DINÁMICA DE HARDWARE
    if torch.cuda.is_available():
        dispositivo = torch.device('cuda')
        torch.backends.cudnn.benchmark = True  
        num_workers_loader = 2                 
        pin_memory_loader = True               
        max_hilos_carga = 6                    
        print("🚀 Pipeline en ejecución: Utilizando NVIDIA GPU (CUDA) calibrada para 4GB VRAM.")
    elif torch.backends.mps.is_available():
        dispositivo = torch.device('mps')
        num_workers_loader = 0
        pin_memory_loader = False
        max_hilos_carga = 4
        print("🚀 Pipeline en ejecución: Utilizando Apple Silicon GPU (MPS).")
    else:
        dispositivo = torch.device('cpu')
        num_workers_loader = 0
        pin_memory_loader = False
        max_hilos_carga = 2                    
        print("💻 Pipeline en ejecución: Utilizando CPU.")

    # 2. CARGA MULTIHILO DE ARCHIVOS EN MEMORIA
    orquestador = StepManiaDataOrchestrator(DIRECTORIO_PACKS, seq_len=SEQ_LEN, n_mels=N_MELS)
    ds_train, ds_test = orquestador.construir_datasets_clasificados(
        test_size=0.2, 
        duplicar_train_aug=True,
        max_workers=max_hilos_carga  
    )

    if ds_train and len(ds_train) > 0:
        # 3. DATALOADERS OPTIMIZADOS
        loader_train = DataLoader(
            ds_train, 
            batch_size=BATCH_SIZE, 
            shuffle=True,
            num_workers=num_workers_loader,
            pin_memory=pin_memory_loader
        )
        loader_test = DataLoader(
            ds_test, 
            batch_size=BATCH_SIZE, 
            shuffle=False,
            num_workers=num_workers_loader,
            pin_memory=pin_memory_loader
        )

        # 4. INICIALIZACIÓN DEL MODELO
        modelo = StepTransformerMulticanalCausal(
            vocab_size=VOCAB_SIZE, 
            seq_len=SEQ_LEN, 
            input_dims=INPUT_DIMS
        ).to(dispositivo)

        criterio = nn.CrossEntropyLoss()
        optimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)

        ckpt_manager = StepManiaCheckpointManager()
        epoch_ini, mejor_loss = ckpt_manager.cargar_checkpoint_si_existe(modelo, optimizador)

        # 5. BUCLE DE ENTRENAMIENTO CON ACUMULACIÓN DE GRADIENTES
        pbar_epochs = tqdm(range(epoch_ini, EPOCHS), desc="Progreso Total", unit="epoch")
        for epoch in pbar_epochs:
            modelo.train()
            loss_total = 0.0
            optimizador.zero_grad()
            
            pbar_batches = tqdm(loader_train, desc=f"Época {epoch+1}/{EPOCHS}", leave=False, unit="batch")
            for idx_b, (X_b, Y_b) in enumerate(pbar_batches):
                X_b, Y_b = X_b.to(dispositivo, non_blocking=pin_memory_loader), Y_b.to(dispositivo, non_blocking=pin_memory_loader)
                
                outputs = modelo(X_b)
                loss = criterio(outputs.view(-1, VOCAB_SIZE), Y_b.view(-1))
                loss_escalada = loss / PASOS_ACUMULACION
                
                loss_escalada.backward()
                
                if (idx_b + 1) % PASOS_ACUMULACION == 0 or (idx_b + 1) == len(loader_train):
                    optimizador.step()
                    optimizador.zero_grad()
                
                loss_total += loss.item()
                pbar_batches.set_postfix({"loss_inst": f"{loss.item():.4f}"})
            
            loss_promedio = loss_total / len(loader_train)
            tqdm.write(f" Epoch [ ✨ {epoch+1}/{EPOCHS}] Fin. Loss promedio: {loss_promedio:.4f}")
            
            mejor_loss = ckpt_manager.guardar_mejor_modelo(modelo, optimizador, epoch, loss_promedio, mejor_loss)
        
        ckpt_manager.guardar_worse_al_finalizar()

        # 6. VALIDACIÓN EN RITMO (INFERENCIA)
        if len(ds_test) > 0:
            print(" Ejecutando validación visual interactiva en el conjunto de Test... ⚡")
            modelo.eval()
            tokenizer = ds_test.tokenizer
            X_sample, Y_sample = ds_test[0]
            
            X_sample = X_sample.unsqueeze(0).to(dispositivo)
            with torch.no_grad():
                logits_sample = modelo(X_sample)
                preds_sample = torch.argmax(logits_sample, dim=-1).squeeze(0).cpu().numpy()
            
            lista_real = [tokenizer.decode_id(token_id) for token_id in Y_sample.numpy()]
            lista_pred = [tokenizer.decode_id(token_id) for token_id in preds_sample]
            VisualStepValidator.imprimir_comparativa(lista_real, lista_pred, max_pasos=20)
    else:
        print(f"[!] Sistema listo. Añade archivos .sm/.ssc/.sma y música en '{DIRECTORIO_PACKS}' para poblar los subconjuntos.")
