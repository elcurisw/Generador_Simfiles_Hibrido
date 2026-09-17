import torch  # <-- DEBE SER LA LÍNEA 1
import torch.nn as nn
import os
import json
import random
import librosa
import itertools
import shutil
import threading
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
import customtkinter as ctk
from tkinter import filedialog, messagebox

MAX_LEVEL = 30
checkpoint_dir="./checkpoints"
model_name="step_transformer_model.pt"

def fijar_semilla_determinista(seed_value):
    """Fija todas las semillas aleatorias para garantizar reproducibilidad exacta."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# =====================================================================
# 1. CORE DE IA
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

class StepTransformerMulticanalCausal(nn.Module):
    def __init__(self, vocab_size, seq_len, input_dims=66, d_model=128, nhead=4, 
                 num_layers=4, dim_feedforward=256):
        super().__init__()
        self.seq_len = seq_len
        self.proyeccion_audio = nn.Linear(input_dims, d_model)
        self.pos_embedding = nn.Embedding(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
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

class PostProcesadorStepMania:
    @staticmethod
    def corregir_sintaxis_holds(secuencia_pasos, pasos_maximos_hold=12, lineas_por_compas=16, max_holds_simultaneos=2):
        # ---------------------------------------------------------------------
        # PASO A: DETECCIÓN DE REPETICIONES CONTINUAS (FUSIÓN A ROLLS '4')
        # ---------------------------------------------------------------------
        pasos_modificados = [list(p) for p in secuencia_pasos]
        total_pasos = len(pasos_modificados)

        # ---------------------------------------------------------------------
        # PASO B: DETECCIÓN DE REPETICIONES CONTINUAS (FUSIÓN A ROLLS '4')
        # ---------------------------------------------------------------------
        UMBRAL_REPETICION = 3  # Activa la fusión al detectar 3 o más notas consecutivas
        
        for col in range(4):
            i = 0
            while i < total_pasos:
                if pasos_modificados[i][col] == '1':
                    inicio_rafaga = i
                    longitud = 0
                    ultimo_indice_nota = i
                    
                    while i < total_pasos and (pasos_modificados[i][col] == '1' or pasos_modificados[i][col] == '0'):
                        if pasos_modificados[i][col] == '1':
                            longitud += 1
                            ultimo_indice_nota = i
                        if i - ultimo_indice_nota > 1:
                            break
                        i += 1
                    
                    if longitud >= UMBRAL_REPETICION:
                        fin_real = min(ultimo_indice_nota, inicio_rafaga + pasos_maximos_hold)
                        pasos_modificados[inicio_rafaga][col] = '4' # Cabeza de Roll
                        for k in range(inicio_rafaga + 1, fin_real):
                            if pasos_modificados[k][col] in ['1', '3']:
                                pasos_modificados[k][col] = '0' # Limpieza del cuerpo intermedio
                        pasos_modificados[fin_real][col] = '3' # Cierre del Roll
                        i = fin_real + 1
                        continue
                i += 1

        # ---------------------------------------------------------------------
        # PASO B: VALIDACIÓN SINTÁCTICA ESTRICTA (SOPORTE PARA HOLDS '2' Y ROLLS '4')
        # ---------------------------------------------------------------------
        secuencia_limpia = []
        contadores_sostenidos = [-1, -1, -1, -1] 
        
        for i, paso in enumerate(pasos_modificados):
            nuevo_paso = list(paso)
            for col in range(4):
                if contadores_sostenidos[col] >= 0:
                    contadores_sostenidos[col] += 1

            for col in range(4):
                char = nuevo_paso[col]
                if contadores_sostenidos[col] >= 0:
                    if char == '3' or contadores_sostenidos[col] >= pasos_maximos_hold:
                        nuevo_paso[col] = '3'
                        contadores_sostenidos[col] = -1
                    else:
                        nuevo_paso[col] = '0'
                else:
                    if char in ['2', '4']: # Validar tanto cabezas de Hold como de Roll
                        sostenidos_activos = [c for c in range(4) if contadores_sostenidos[c] >= 0]
                        if len(sostenidos_activos) >= max_holds_simultaneos:
                            nuevo_paso[col] = '1'
                        else:
                            contadores_sostenidos[col] = 0
                    elif char == '3':
                        nuevo_paso[col] = '0'

            # ---------------------------------------------------------------------
            # PASO C: FILTRADO ANATÓMICO (MÁXIMO 2 INTERACCIONES SIMULTÁNEAS)
            # ---------------------------------------------------------------------
            flechas_impacto = [idx for idx, char in enumerate(nuevo_paso) if char in ['1', '2', '4']]
            cuerpos_activos = [idx for idx, v in enumerate(contadores_sostenidos) if v > 0]
            total_peso_anatomico = len(flechas_impacto) + len(cuerpos_activos)
            
            if total_peso_anatomico > 2:
                exceso = total_peso_anatomico - 2
                flechas_simples = [idx for idx in flechas_impacto if nuevo_paso[idx] == '1']
                remover = random.sample(flechas_simples, min(len(flechas_simples), exceso))
                for idx in remover:
                    nuevo_paso[idx] = '0'

            secuencia_limpia.append("".join(nuevo_paso))

        # ---------------------------------------------------------------------
        # PASO D: CIERRE DE SEGURIDAD GENERAL
        # ---------------------------------------------------------------------
        if any(c >= 0 for c in contadores_sostenidos):
            ultimo_paso_lista = list(secuencia_limpia[-1])
            for col in range(4):
                if contadores_sostenidos[col] >= 0: 
                    ultimo_paso_lista[col] = '3'
            secuencia_limpia[-1] = "".join(ultimo_paso_lista)
            
        return secuencia_limpia
    
    @staticmethod
    def inyectar_secciones_saltos(secuencia_pasos, lista_rms, rms_medio, duracion, bpm, lineas_por_compas):
        """
        Detecta zonas de alta energía y convierte notas simples en saltos.
        Calcula dinámicamente la fuerza de borrado según los BPM y la resolución del compás.
        Limpia las líneas subsecuentes de manera segura (incluyendo holders completos).
        """
        if len(lista_rms) == 0:
            return secuencia_pasos

        pasos_modificados = [list(p) for p in secuencia_pasos]
        total_pasos = len(pasos_modificados)
        segundos_por_paso = ((60.0 / bpm) * 4.0) / lineas_por_compas
        total_frames = len(lista_rms)

        # ---------------------------------------------------------------------
        # CÁLCULO DINÁMICO DE LA FUERZA DE BORRADO (Rango objetivo: ~100ms a ~200ms de espacio)
        # ---------------------------------------------------------------------
        # Calculamos cuántas líneas equivalen a una fracción de tiempo lógica según el tempo
        if bpm < 100:
            # En canciones lentas, un borrado pequeño es suficiente
            proporcion_borrado = 0.125  # Equivalente a 1/32 de compás estándar
        elif bpm > 180:
            # En canciones muy rápidas, necesitamos dar más aire visual (más líneas)
            proporcion_borrado = 0.250  # Equivalente a 1/16 de compás estándar
        else:
            # Rango medio balanceado
            proporcion_borrado = 0.187  

        # Escalamos el número de pasos a borrar según la resolución del compás actual (8, 12, 16, etc.)
        pasos_a_borrar = max(1, int(round(lineas_por_compas * proporcion_borrado)))

        i = 0
        while i < total_pasos:
            segundo_actual = i * segundos_por_paso
            frame_idx = min(int((segundo_actual / duracion) * total_frames), total_frames - 1)
            ratio_energia = lista_rms[frame_idx] / rms_medio if rms_medio > 0 else 1.0

            # Activar únicamente en picos rítmicos de alta energía (Drops)
            if ratio_energia > 1.25:
                paso_actual = pasos_modificados[i]
                
                # Modificar solo si la línea actual tiene una única nota simple libre (evita romper secuencias ya complejas)
                if paso_actual.count('1') == 1 and paso_actual.count('2') == 0 and paso_actual.count('4') == 0:
                    columnas_vacias = [col for col, char in enumerate(paso_actual) if char == '0']
                    if columnas_vacias:
                        col_nueva = random.choice(columnas_vacias)
                        paso_actual[col_nueva] = '1'
                        pasos_modificados[i] = paso_actual
                        
                        # Ejecutar el borrado seguro con la fuerza calculada dinámicamente
                        for k in range(1, pasos_a_borrar + 1):
                            target_idx = i + k
                            if target_idx < total_pasos:
                                for col in range(4):
                                    char_target = pasos_modificados[target_idx][col]
                                    
                                    # CASO A: Nota simple común -> Se borra directamente
                                    if char_target == '1':
                                        pasos_modificados[target_idx][col] = '0'
                                        
                                    # CASO B: Cabeza de Hold ('2') o Roll ('4') -> Borrado en cascada
                                    elif char_target in ['2', '4']:
                                        pasos_modificados[target_idx][col] = '0'
                                        # Buscamos hacia adelante en la misma columna el cierre '3' para fulminarlo
                                        for scan_forward in range(target_idx + 1, total_pasos):
                                            if pasos_modificados[scan_forward][col] == '3':
                                                pasos_modificados[scan_forward][col] = '0'
                                                break
                                            elif pasos_modificados[scan_forward][col] == '0':
                                                continue
                                            else:
                                                # Protección perimetral por si la estructura está corrupta de origen
                                                break
                                                
                                    # CASO C: Cuerpo intermedio o final '3' de un hold que empezó ANTES del salto.
                                    # NO lo tocamos para mantener la coherencia del juego y que el pie siga apoyado.
                                    elif char_target == '3':
                                        continue
                        
                        # Desplazamos el puntero el equivalente al salto + el espacio borrado
                        i += pasos_a_borrar + 1
                        continue
            i += 1

        return ["".join(p) for p in pasos_modificados]

    @staticmethod
    def recalcular_meter_real(secuencia_pasos, duracion_segundos, dificultad_tag, max_level_chosen):
        """
        Calcula el METER real basado en picos de densidad (NPS) y modificadores avanzados.
        Ajusta proporcionalmente las dificultades respetando el techo máximo elegido en la UI.
        """
        if not secuencia_pasos or duracion_segundos <= 0:
            return 1, ""
            
        total_impactos = 0
        total_modificadores = 0
        
        # Analizar densidad general y presencia de trampas
        for paso in secuencia_pasos:
            total_impactos += sum(1 for char in paso if char in ['1', '2', '4'])
            total_modificadores += sum(1 for char in paso if char in ['M', 'F', 'L', 'S', 'H', 'D'])
            
        nps_promedio = total_impactos / duracion_segundos
        
        # Factor de peligro extra por presencia de trampas (Cada 10 trampas suben un poco la tensión)
        factor_peligro_trampas = min(2.0, (total_modificadores / (total_impactos + 1)) * 5.0)

        # ESCALADO PROPORCIONAL DINÁMICO BASADO EN EL SLIDER DE LA INTERFAZ
        # Calculamos los techos de nivel exactos que espera ver la GUI
        techo_easy = max(1, int(max_level_chosen * 0.25))
        techo_med  = max(3, int(max_level_chosen * 0.50))
        techo_hard = max(6, int(max_level_chosen * 0.75))
        techo_chal = max(12, max_level_chosen)

        if dificultad_tag == "Easy":
            meter_estimado = int(nps_promedio * 2.5) + 1
            meter_real = max(1, min(meter_estimado, techo_easy))
        elif dificultad_tag == "Medium":
            meter_estimado = int(nps_promedio * 3.0) + 2 + int(factor_peligro_trampas)
            meter_real = max(techo_easy + 1, min(meter_estimado, techo_med))
        elif dificultad_tag == "Hard":
            # En niveles altos sumamos el factor de peligro de forma flotante antes del casteo
            meter_estimado = int((nps_promedio * 3.5) + 3 + factor_peligro_trampas)
            meter_real = max(techo_med + 1, min(meter_estimado, techo_hard))
        else: # Challenge
            meter_estimado = int((nps_promedio * 4.2) + 4 + (factor_peligro_trampas * 1.5))
            # Permite liberar el potencial si la canción supera las expectativas
            meter_real = max(techo_hard + 1, min(meter_estimado, techo_chal))

        # Generar barra visual adaptada al techo máximo real del programa (30 bloques)
        bloques_barra = "█" * meter_real
        espacios_barra = "░" * (30 - meter_real)
        
        # Construir reporte detallado para la consola verde de la GUI
        reporte = (
            f"📊 [{dificultad_tag.upper()}] NPS Promedio: {nps_promedio:.2f} | Notas: {total_impactos}\n"
            f"⚠️ Modificadores/Trampas detectadas: {total_modificadores}\n"
            f"🎯 METER DINÁMICO: Nivel {meter_real} (Techo Máx de Escala: {techo_chal})\n"
            f"└─ [{bloques_barra}{espacios_barra}]\n"
            f"{'-'*45}\n"
        )

        return meter_real, reporte


# =====================================================================
# 2. MOTOR DSP ESPECTRAL
# =====================================================================
def analizar_audio_hibrido(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if hasattr(tempo, "__len__"): 
            bpm_detectado = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            bpm_detectado = float(tempo)
        if bpm_detectado < 40 or bpm_detectado > 250:
            bpm_detectado = 120.0
        duracion_segundos = librosa.get_duration(y=y, sr=sr)
        rms = librosa.feature.rms(y=y).flatten()
        rms_medio = float(rms.mean()) if len(rms) > 0 else 1.0
        centroide = librosa.feature.spectral_centroid(y=y, sr=sr).flatten()
        centroide_medio = float(centroide.mean()) if len(centroide) > 0 else 1.0
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        return bpm_detectado, duracion_segundos, rms, rms_medio, centroide, centroide_medio, mel_db
    except Exception as e:
        print(f"Error en análisis DSP: {e}")
        return 120.0, 180.0, np.array([]), 1.0, np.array([]), 1.0, None

def detectar_primer_drop_rms(rms, sr, hop_length=512, umbral_porcentaje=0.05):
    """
    MODIFICADO: Detecta el segundo exacto del impacto rítmico inicial 
    con sensibilidad mejorada mediante un umbral adaptativo (RMS dinámico).
    """
    if len(rms) == 0:
        return 0.000

    # 1. Establecer el suelo de ruido analizando los frames iniciales más calmados
    suelo_ruido = np.percentile(rms, 15)  
    rms_maximo = np.max(rms)
    
    # 2. El umbral combina el porcentaje del pico máximo y el suelo base del archivo
    umbral_energia = suelo_ruido + (rms_maximo - suelo_ruido) * umbral_porcentaje
    
    # 3. Aplicar filtro de gradiente (Derivada) para buscar subidas abruptas de energía
    # Esto evita falsos positivos causados por introducciones con un sutil Fade-In
    rms_diff = np.diff(rms, prepend=0)
    umbral_cambio = np.std(rms_diff) * 0.5  # Sensibilidad al cambio de transitorios
    
    # Encontrar el primer frame que cumple con volumen suficiente y crecimiento rítmico
    indices_candidatos = np.where((rms > umbral_energia) & (rms_diff > umbral_cambio))[0]
    
    if len(indices_candidatos) > 0:
        primer_frame = indices_candidatos[0]
        tiempo_segundos = librosa.frames_to_time(primer_frame, sr=sr, hop_length=hop_length)
        return round(float(tiempo_segundos), 3)
        
    # Caída de seguridad pasiva por si fallan los diferenciales de gradiente
    indices_pasivos = np.where(rms > umbral_energia)[0]
    if len(indices_pasivos) > 0:
        primer_frame = indices_pasivos[0]
        tiempo_segundos = librosa.frames_to_time(primer_frame, sr=sr, hop_length=hop_length)
        return round(float(tiempo_segundos), 3)

    return 0.000

# =====================================================================
# 3. CORE DE GENERACIÓN HÍBRIDA DUAL (.SM y .SSC)
# =====================================================================
def generar_simfiles_hibridos(audio_path, checkpoint_path, song_title, max_level_chosen, 
                              carpeta_salida, artist_name="", banner_path="", video_path="", duracion_limite=0.0, custom_params=None):

    # A. Recuperar y fijar la semilla de forma estricta antes de que la IA o el DSP hagan algo
    seed_actual = custom_params.get("seed_value", 42)
    fijar_semilla_determinista(seed_actual)

    config_dificultad = {
        "Easy": {
            "temperatura": custom_params["temperatura"] if custom_params else 0.7, 
            "max_hold": custom_params["max_hold"] if custom_params else 4, 
            "meter": max(1, int(max_level_chosen * 0.25)),
            "lineas_por_compas": 8, 
            # Nuevos tipos de notas (Desactivados en Easy por coherencia rítmica)
            "max_minas_compas": 0, "probabilidad_minas": 0.0,
            "max_fakes_compas": 0, "probabilidad_fakes": 0.0,
            "max_lifts_compas": 0, "probabilidad_lifts": 0.0,
            "max_potions_compas": 0, "probabilidad_potions": 0.0,
            "max_shields_compas": 0, "probabilidad_shields": 0.0,
            "max_rayos_compas": 0, "probabilidad_rayos": 0.0,
            "max_hiddens_compas": 0, "probabilidad_hiddens": 0.0,
        },
        "Medium": {
            "temperatura": custom_params["temperatura"] if custom_params else 1.0, 
            "max_hold": custom_params["max_hold"] if custom_params else 4, 
            "meter": max(2, int(max_level_chosen * 0.50)),
            "lineas_por_compas": 8, 
            "max_minas_compas": min(1, custom_params["max_minas_compas"]) if custom_params else 1, 
            "probabilidad_minas": custom_params["probabilidad_minas"] if custom_params else 0.15,
            "max_fakes_compas": min(1, custom_params["max_fakes_compas"]) if custom_params else 1, 
            "probabilidad_fakes": (custom_params["probabilidad_fakes"] * 0.5) if custom_params else 0.07,
            "max_lifts_compas": min(1, custom_params["max_lifts_compas"]) if custom_params else 1, 
            "probabilidad_lifts": (custom_params["probabilidad_lifts"] * 0.5) if custom_params else 0.07,
            "max_potions_compas": min(1, custom_params["max_potions_compas"]) if custom_params else 1,  
            "probabilidad_potions": (custom_params["probabilidad_potions"] * 0.3) if custom_params else 0.05,
            "max_shields_compas": min(1, custom_params["max_shields_compas"]) if custom_params else 1,  
            "probabilidad_shields": (custom_params["probabilidad_shields"] * 0.3) if custom_params else 0.05,
            "max_rayos_compas": min(1, custom_params["max_rayos_compas"]) if custom_params else 1,  
            "probabilidad_rayos": (custom_params["probabilidad_rayos"] * 0.3) if custom_params else 0.05,
            "max_hiddens_compas": min(1, custom_params["max_hiddens_compas"]) if custom_params else 1,  
            "probabilidad_hiddens": (custom_params["probabilidad_hiddens"] * 0.4) if custom_params else 0.06,
        },
        "Hard": {
            "temperatura": custom_params["temperatura"] if custom_params else 1.15, 
            "max_hold": custom_params["max_hold"] if custom_params else 4, 
            "meter": max(3, int(max_level_chosen * 0.75)),
            "lineas_por_compas": 12, "min_notas_compas": 4, "max_notas_compas": 9, 
            "max_minas_compas": min(2, custom_params["max_minas_compas"]) if custom_params else 2, 
            "probabilidad_minas": custom_params["probabilidad_minas"] if custom_params else 0.25,
            "max_fakes_compas": min(2, custom_params["max_fakes_compas"]) if custom_params else 1, 
            "probabilidad_fakes": (custom_params["probabilidad_fakes"] * 0.8) if custom_params else 0.20,
            "max_lifts_compas": min(2, custom_params["max_lifts_compas"]) if custom_params else 1, 
            "probabilidad_lifts": (custom_params["probabilidad_lifts"] * 0.8) if custom_params else 0.20,
            "max_potions_compas": min(2, (custom_params["max_potions_compas"])) if custom_params else 1, 
            "probabilidad_potions": (custom_params["probabilidad_potions"] * 0.5) if custom_params else 0.12,
            "max_shields_compas": min(2, (custom_params["max_shields_compas"])) if custom_params else 1, 
            "probabilidad_shields": (custom_params["probabilidad_shields"] * 0.5) if custom_params else 0.12,
            "max_rayos_compas": min(2, (custom_params["max_rayos_compas"])) if custom_params else 1, 
            "probabilidad_rayos": (custom_params["probabilidad_rayos"] * 0.5) if custom_params else 0.12,
            "max_hiddens_compas": min(2, custom_params["max_hiddens_compas"]) if custom_params else 1, 
            "probabilidad_hiddens": (custom_params["probabilidad_hiddens"] * 0.7) if custom_params else 0.17,
        },
        "Challenge": {
            "temperatura": custom_params["temperatura"] if custom_params else 1.3, 
            "max_hold": custom_params["max_hold"] if custom_params else 4, 
            "meter": max_level_chosen,
            "lineas_por_compas": 12, "min_notas_compas": 8, "max_notas_compas": 12, 
            "max_minas_compas": custom_params["max_minas_compas"] if custom_params else 3, 
            "probabilidad_minas": custom_params["probabilidad_minas"] if custom_params else 0.35,
            "max_fakes_compas": custom_params["max_fakes_compas"] if custom_params else 3, 
            "probabilidad_fakes": custom_params["probabilidad_fakes"] if custom_params else 0.35,
            "max_lifts_compas": custom_params["max_lifts_compas"] if custom_params else 3, 
            "probabilidad_lifts": custom_params["probabilidad_lifts"] if custom_params else 0.35,
            "max_potions_compas": custom_params["max_potions_compas"] if custom_params else 2, 
            "probabilidad_potions": (custom_params["probabilidad_potions"] * 0.6) if custom_params else 0.21,
            "max_shields_compas": custom_params["max_shields_compas"] if custom_params else 2, 
            "probabilidad_shields": (custom_params["probabilidad_shields"] * 0.6) if custom_params else 0.21,
            "max_rayos_compas": custom_params["max_rayos_compas"] if custom_params else 2, 
            "probabilidad_rayos": custom_params["probabilidad_rayos"] if custom_params else 0.21,
            "max_hiddens_compas": custom_params["max_hiddens_compas"] if custom_params else 3, 
            "probabilidad_hiddens": custom_params["probabilidad_hiddens"] if custom_params else 0.28,
        }
    }

    bpm, duracion, lista_rms, rms_medio, lista_centroide, centroide_medio, mel_db = analizar_audio_hibrido(audio_path)
    
    if duracion_limite > 0.0:
        duracion = min(duracion, duracion_limite)
    
    double_bpm_factor = 2.0 if (custom_params and custom_params.get("double_bpm") == True) else 1.0

    #Aplicamos un bpm manual si es configurado desde la interfaz
    bpm_manual = custom_params.get("bpm_manual")
    if bpm_manual >= 60:
        bpm = bpm_manual

    if double_bpm_factor > 1.0:
        bpm *= double_bpm_factor

    # Extraer el offset numérico de la interfaz gráfica
    # --- CÁLCULO DE OFFSET AUTOMÁTICO VS MANUAL ---
    if custom_params.get("offset_automatico", False):
        # Detectamos el offset real analizando el RMS. 
        # Librosa por defecto usa un hop_length de 512 en librosa.feature.rms
        val_offset = detectar_primer_drop_rms(lista_rms, sr=22050, hop_length=512)
    else:
        val_offset = custom_params["offset_manual"] if custom_params else 0.000

    tokenizer = StepTokenizer()
    dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelo = StepTransformerMulticanalCausal(vocab_size=628, seq_len=128, input_dims=66).to(dispositivo)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=dispositivo)
        modelo.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise FileNotFoundError(f"Checkpoint no encontrado en: {checkpoint_path}")
    modelo.eval()

    pasos_por_segundo = (bpm / 60.0) * 4.0
    total_frames = len(lista_rms)
    mapa_pasos_por_dificultad = {}

    compases_totales = math.ceil((duracion * (bpm / 60.0)) / 4.0)

    # Validar booleano de posprocesamiento enviado por la interfaz
    aplicar_post = custom_params.get("aplicar_postprocesamiento", True) if custom_params else True

    # --- LÓGICA EXCLUYENTE DE BPMS Y SPEEDS DINÁMICOS CON LECTURA DE SLIDERS ---
    dinamico_activo = custom_params.get("aplicar_bpm_dinamico", False) if custom_params else False
    speeds_activo = custom_params.get("aplicar_speeds_dinamicos", False) if custom_params else False

    bpms_string_line = f"0.000={bpm:.3f}"
    speeds_string_line = "" 

    # Se leen los límites asignados por el usuario desde los sliders de la interfaz
    VELOCIDAD_MINIMA = custom_params.get("speed_min_custom", 0.70) if custom_params else 0.70
    VELOCIDAD_MAXIMA = custom_params.get("speed_max_custom", 1.40) if custom_params else 1.40

    if dinamico_activo and not speeds_activo:
        # CASO 1: Altera el BPM real usando la lista rítmica base
        modificadores_bloque = [1.0, 1.0, 1.25, 1.0, 0.75, 1.5, 1.0]
        cambios_bpm = []
        for compas_i in range(0, compases_totales, 8):
            beat_inicial = float(compas_i * 4)
            mod_actual = modificadores_bloque[(compas_i // 8) % len(modificadores_bloque)]
            bpm_calculado = (bpm / double_bpm_factor) * mod_actual * double_bpm_factor
            cambios_bpm.append(f"{beat_inicial:.3f}={bpm_calculado:.3f}")
        bpms_string_line = ",\n".join(cambios_bpm)
    elif speeds_activo and not dinamico_activo:
            # CASO 2: BPM estático con velocidades visuales progresivas basadas en RMS
            cambios_speeds = []
            lineas_objetivo = 8
            segundos_por_paso = ((60.0 / bpm) * 4.0) / lineas_objetivo
            
            # Se lee el tiempo de interpolación desde la interfaz gráfica (Ej: 2.0 Beats)
            duracion_transicion = custom_params.get("speed_trans_custom", 0.0) if custom_params else 0.0
            
            # SE LEEN LOS LÍMITES DE SENSIBILIDAD RMS DE LA INTERFAZ
            r_min = custom_params.get("ratio_min_energia", 0.5) if custom_params else 0.5
            r_max = custom_params.get("ratio_max_energia", 1.5) if custom_params else 1.5
            # Asegurar que no ocurra división por cero si los sliders tienen el mismo valor
            r_diff = (r_max - r_min) if (r_max - r_min) != 0 else 1.0

            for compas_i in range(0, compases_totales, 8):
                beat_inicial = float(compas_i * 4)
                paso_inicial_seccion = compas_i * lineas_objetivo
                paso_final_seccion = (compas_i + 8) * lineas_objetivo
                frames_de_la_seccion = []
                
                for p_idx in range(paso_inicial_seccion, paso_final_seccion):
                    seg_act = p_idx * segundos_por_paso
                    f_idx = min(int((seg_act / duracion) * total_frames), total_frames - 1)
                    if len(lista_rms) > 0:
                        frames_de_la_seccion.append(lista_rms[f_idx])
                        
                rms_local = np.mean(frames_de_la_seccion) if frames_de_la_seccion else rms_medio
                ratio_energia = rms_local / rms_medio if rms_medio > 0 else 1.0
                
                # REAJUSTE CON LA SENSIBILIDAD RMS SELECCIONADA POR EL USUARIO:
                # Normaliza la energía actual entre el rango r_min y r_max (clamped entre 0 y 1)
                factor_intensidad = (ratio_energia - r_min) / r_diff
                factor_intensidad = max(0.0, min(factor_intensidad, 1.0))
                
                # Interpola linealmente entre la velocidad visual mínima y máxima configurada
                ratio_velocidad = VELOCIDAD_MINIMA + factor_intensidad * (VELOCIDAD_MAXIMA - VELOCIDAD_MINIMA)
                
                # 1. Entrada progresiva (Transición suave configurada por el usuario)
                cambios_speeds.append(f"{beat_inicial:.3f}={ratio_velocidad:.3f}={duracion_transicion:.3f}=0")
                
                # 2. SISTEMA DE REAJUSTE SEGURO
                # Si hay una transición activa, fuerza un evento de fijación estricta al terminar la curva
                if duracion_transicion > 0.0:
                    beat_reajuste = beat_inicial + duracion_transicion
                    # Impide que el reajuste invada el siguiente bloque de 8 compases (32 beats)
                    if beat_reajuste < float((compas_i + 8) * 4):
                        cambios_speeds.append(f"{beat_reajuste:.3f}={ratio_velocidad:.3f}=0.000=0")

            speeds_string_line = ",\n".join(cambios_speeds)

    log_metricas_diff = ""

    for diff, cfg in config_dificultad.items():
        pasos_crudos_ia = []
        lineas_objetivo = cfg["lineas_por_compas"]
        total_pasos_dificultad = compases_totales * lineas_objetivo
        segundos_por_paso = ((60.0 / bpm) * 4.0) / lineas_objetivo

        for paso_idx in range(total_pasos_dificultad):
            segundo_actual = paso_idx * segundos_por_paso

            # --- MODIFICACIÓN DEL OFFSET: SILENCIO FORZADO HASTA CUMPLIR EL TIEMPO ---
            if val_offset > 0.0 and segundo_actual < val_offset:
                pasos_crudos_ia.append("0000") # No crear nada hasta que se complete el offset
                continue

            frame_idx = min(int((segundo_actual / duracion) * total_frames), total_frames - 1)

            if paso_idx > 0 and pasos_crudos_ia[-1] != "0000":
                limite_densidad = {"Easy": 0.40, "Medium": 0.30, "Hard": 0.15, "Challenge": 0.05}
                if random.random() < limite_densidad[diff]:
                    pasos_crudos_ia.append("0000")
                    continue

            bloque_contexto = []
            for offset in range(-127, 1):
                f = max(0, min(frame_idx + offset, total_frames - 1))
                r_energia = lista_rms[f] / rms_medio if len(lista_rms) > 0 else 1.0
                r_brillo = lista_centroide[f] / centroide_medio if len(lista_centroide) > 0 else 1.0
                vector_mel = mel_db[:, f].tolist() if mel_db is not None else [0.0]*64
                bloque_contexto.append([r_energia, r_brillo] + vector_mel)

            tensor_in = torch.tensor(bloque_contexto, dtype=torch.float32).unsqueeze(0).to(dispositivo)
            with torch.no_grad():
                logits = modelo(tensor_in)[:, -1, :]
                probabilidades = torch.softmax(logits / cfg["temperatura"], dim=-1)
                prediccion = torch.multinomial(probabilidades, num_samples=1).item()

            if paso_idx < 2:
                pasos_crudos_ia.append("1000")
            else:
                pasos_crudos_ia.append("".join(tokenizer.decode_id(prediccion)))

        # === Ampliar la duración estética al final ===
        extension_final = custom_params.get("extension_final", 0.0) if custom_params else 0.0
        # === INYECCIÓN SEGURA DE NOTAS DE EXTENSIÓN FINAL ===
        # Esto añade compases vacíos y una nota final de tipo "Hold" (2...3) o nota simple (1) 
        # que no afecta el juego ya que la música habrá terminado.
        if extension_final > 0:
            segundos_por_compas = (60.0 / bpm) * 4.0
            compases_extras = math.ceil(extension_final / segundos_por_compas)
            lineas_extras = compases_extras * lineas_objetivo
            for l_idx in range(lineas_extras):
                if l_idx == lineas_extras - 1:
                    # Cierre obligatorio del Hold en la última línea extra
                    pasos_crudos_ia.append("V000") 
                else:
                    # Líneas vacías intermedias
                    pasos_crudos_ia.append("0000")

        # --- SELECCIÓN DEL FORMATO DE PASOS ---
        if aplicar_post:
            limite_holds = custom_params["holds_simultaneos"] if custom_params else 2
            pasos_finales = PostProcesadorStepMania.corregir_sintaxis_holds(
                pasos_crudos_ia, 
                pasos_maximos_hold=cfg["max_hold"], 
                lineas_por_compas=lineas_objetivo,
                max_holds_simultaneos=limite_holds,
            )
            # === INYECCIÓN DE SECCIONES DE SALTOS ===
            if custom_params.get("activar_secciones_saltos", False):
                pasos_finales = PostProcesadorStepMania.inyectar_secciones_saltos(
                    secuencia_pasos=pasos_finales,
                    lista_rms=lista_rms,
                    rms_medio=rms_medio,
                    duracion=duracion,
                    bpm=bpm,
                    lineas_por_compas=lineas_objetivo
                )
        else:
            pasos_finales = pasos_crudos_ia  # Pasos intactos tal como salieron del modelo

        # --- SISTEMA CONDICIONAL DE REESCRITURA DE DIFICULTAD REAL ---
        if custom_params.get("recalcular_dificultad", False):
            meter_real, texto_reporte = PostProcesadorStepMania.recalcular_meter_real(
                secuencia_pasos=pasos_finales, 
                duracion_segundos=duracion, 
                dificultad_tag=diff,
                max_level_chosen=max_level_chosen
            )
            config_dificultad[diff]["meter"] = meter_real
            log_metricas_diff += texto_reporte

        bloque_pasos_texto = ""
        compas = []
        for idx, paso in enumerate(pasos_finales):
            compas.append(paso)

            if len(compas) == lineas_objetivo or idx == len(pasos_finales) - 1:
                # Inicialización segura de rms_local antes de cualquier condición
                rms_local = rms_medio 

                if "min_notas_compas" in cfg and "max_notas_compas" in cfg:
                    paso_inicial_compas = idx - len(compas) + 1
                    frames_del_compas = []
                    for p_idx in range(paso_inicial_compas, idx + 1):
                        seg_act = p_idx * segundos_por_paso
                        f_idx = min(int((seg_act / duracion) * total_frames), total_frames - 1)
                        if len(lista_rms) > 0: frames_del_compas.append(lista_rms[f_idx])
                    
                    rms_local = np.mean(frames_del_compas) if frames_del_compas else rms_medio
                    ratio_actual = rms_local / rms_medio if rms_medio > 0 else 1.0
                    
                    r_min = custom_params["ratio_min_energia"] if custom_params else 0.5
                    r_max = custom_params["ratio_max_energia"] if custom_params else 1.5
                    ratio_clamped = max(r_min, min(ratio_actual, r_max))
                    
                    factor_intensidad = (ratio_clamped - r_min) / (r_max - r_min)
                    limite_permitido = int(math.floor(cfg["min_notas_compas"] + (factor_intensidad * (cfg["max_notas_compas"] - cfg["min_notas_compas"]))))
                    
                    # Se añade el '4' a la detección para mantener la coherencia anatómica
                    indices_con_nota = [i for i, p in enumerate(compas) if any(c in ['1', '2', '4'] for c in p)]
                    if len(indices_con_nota) > limite_permitido:
                        exceso = len(indices_con_nota) - limite_permitido
                        indices_a_eliminar = random.sample(indices_con_nota, exceso)
                        for ind in indices_a_eliminar:
                            # Al limpiar un exceso por energía, se borra el elemento '4' de ser necesario
                            compas[ind] = "".join(['0' if c in ['1', '2', '4'] else c for c in compas[ind]])

                
                # --- LÓGICA DE INSERCIÓN MUTADA CORREGIDA (INTERRUPTOR GLOBAL RMS) ---
                tipos_modificadores = [
                    {"tag": "minas", "char": "M", "max": cfg.get("max_minas_compas", 0), "prob": cfg.get("probabilidad_minas", 0.0), "rms_check": custom_params.get("efectos_por_rms", False)},
                    {"tag": "fakes", "char": "F", "max": cfg.get("max_fakes_compas", 0), "prob": cfg.get("probabilidad_fakes", 0.0), "rms_check": custom_params.get("efectos_por_rms", False)},
                    {"tag": "lifts", "char": "L", "max": cfg.get("max_lifts_compas", 0), "prob": cfg.get("probabilidad_lifts", 0.0), "rms_check": custom_params.get("efectos_por_rms", False)},
                    {"tag": "potions", "char": "P", "max": cfg.get("max_potions_compas", 0), "prob": cfg.get("probabilidad_potions", 0.0), "rms_check": custom_params.get("efectos_por_rms", False)},
                    {"tag": "shields", "char": "D", "max": cfg.get("max_shields_compas", 0), "prob": cfg.get("probabilidad_shields", 0.0), "rms_check": custom_params.get("efectos_por_rms", False)},
                    {"tag": "rayos", "char": "S", "max": cfg.get("max_rayos_compas", 0), "prob": cfg.get("probabilidad_rayos", 0.0), "rms_check": custom_params.get("efectos_por_rms", False)},
                    {"tag": "hiddens", "char": "H", "max": cfg.get("max_hiddens_compas", 0), "prob": cfg.get("probabilidad_hiddens", 0.0), "rms_check": custom_params.get("efectos_por_rms", False)}
                ]

                # Bucle de procesamiento dinámico por RMS con sliders unificados
                for n_type in tipos_modificadores:
                    # VALIDACIÓN CRÍTICA: Si la bandera global de la UI está APAGADA, 
                    # ignoramos por completo la inyección de cualquier trampa o efecto.
                    if not n_type["rms_check"]:
                        continue # Salta este modificador de inmediato sin evaluar nada más

                    max_permitido = n_type["max"]
                    prob_actual = n_type["prob"]
                    
                    if max_permitido > 0:
                        # Escalado dinámico espectral (Sabemos que rms_check es True por el filtro anterior)
                        r_min = custom_params.get("ratio_min_energia", 0.5) if custom_params else 0.5
                        r_max = custom_params.get("ratio_max_energia", 1.5) if custom_params else 1.5
                        r_diff = (r_max - r_min) if (r_max - r_min) != 0 else 1.0
                        
                        factor_intensidad = max(0.0, min((rms_local - r_min) / r_diff, 1.0))
                        prob_actual = min(1.0, prob_actual * (1.0 + factor_intensidad))
                            
                        if random.random() < prob_actual:
                            filas_transformables = [i for i, p in enumerate(compas) if '1' in p]
                            if filas_transformables:
                                if factor_intensidad > 0.75:
                                    cantidad = min(len(filas_transformables), max_permitido)
                                else:
                                    cantidad = min(len(filas_transformables), random.randint(1, max_permitido))
                                    
                                filas_elegidas = random.sample(filas_transformables, cantidad)
                                ultima_columna_mod = -1
                                for ind in filas_elegidas:
                                    fila_lista = list(compas[ind])
                                    indices_flecha = [col for col, char in enumerate(fila_lista) if char == '1']
                                    opciones_filtradas = [c for c in indices_flecha if c != ultima_columna_mod]
                                    columnas_validas = opciones_filtradas if opciones_filtradas else indices_flecha
                                    if columnas_validas:
                                        col_cambio = random.choice(columnas_validas)
                                        fila_lista[col_cambio] = n_type["char"]
                                        compas[ind] = "".join(fila_lista)
                                        ultima_columna_mod = col_cambio
                                
                bloque_pasos_texto += "\n".join(compas)
                bloque_pasos_texto += "\n;\n" if idx == len(pasos_finales) - 1 else "\n,\n"
                compas = []

        mapa_pasos_por_dificultad[diff] = bloque_pasos_texto

    # --- CONTROL DE RECURSOS Y CONTRAMEDIDA SAMEFILEERROR ---
    nombre_audio = os.path.basename(audio_path)
    nombre_banner = os.path.basename(banner_path) if banner_path else ""
    nombre_video = os.path.basename(video_path) if video_path else ""
    
    os.makedirs(carpeta_salida, exist_ok=True)

    # 1. Procesamiento Seguro del Audio
    if custom_params["renombrar_archivos"]:
        _, extension = os.path.splitext(nombre_audio)
        new_audio_name = f"{song_title}{extension}"
    else:
        new_audio_name = nombre_audio
        
    ruta_final_audio = os.path.join(carpeta_salida, new_audio_name)
    if os.path.abspath(audio_path) != os.path.abspath(ruta_final_audio):
        os.rename(audio_path, ruta_final_audio)

    # 2. Procesamiento Seguro del Banner
    new_banner_name = ""
    if banner_path and os.path.exists(banner_path):
        if custom_params["renombrar_archivos"]:
            _, extension = os.path.splitext(nombre_banner)
            new_banner_name = f"{song_title}_banner{extension}"
        else:
            new_banner_name = nombre_banner
            
        ruta_final_banner = os.path.join(carpeta_salida, new_banner_name)
        if os.path.abspath(banner_path) != os.path.abspath(ruta_final_banner):
            shutil.copy(banner_path, ruta_final_banner)
    else:
        new_banner_name = nombre_banner

    # 3. Procesamiento Seguro del Video
    new_video_name = ""
    if video_path and os.path.exists(video_path):
        if custom_params["renombrar_archivos"]:
            _, extension = os.path.splitext(nombre_video)
            new_video_name = f"{song_title}_video{extension}"
        else:
            new_video_name = nombre_video
            
        ruta_final_video = os.path.join(carpeta_salida, new_video_name)
        if os.path.abspath(video_path) != os.path.abspath(ruta_final_video):
            shutil.copy(video_path, ruta_final_video)
    else:
        new_video_name = nombre_video


    bg_changes_line = f"0.000={new_video_name}=1.000=1=0=0=crossfade=," if new_video_name else ""
    val_offset = custom_params["offset_manual"] if custom_params else 0.000

    name_output = f"{song_title}_{seed_actual}"
    output_sm = os.path.join(carpeta_salida, f"{name_output}.sm")
    output_ssc = os.path.join(carpeta_salida, f"{name_output}.ssc")

    # --- ESCRITURA DEL ARCHIVO .SM CLÁSICO ---
    with open(output_sm, "w", encoding="utf-8") as f:
        f.write(f"#TITLE:{song_title};\n#ARTIST:{artist_name};\n#MUSIC:{new_audio_name};\n#BANNER:{new_banner_name};\n")
        f.write(f"#VIDEO:{new_video_name};\n") # Vinculación directa en formato antiguo
        f.write(f"#OFFSET:0.000;\n#BPMS:{bpms_string_line};\n")
        f.write(f"#BGCHANGES:{bg_changes_line};\n\n") # Inyección del script de animación
        
        for diff, bloque in mapa_pasos_por_dificultad.items():
            f.write(f"#NOTES:\n dance-single:\n AI_DSP_Hybrid_Engine:\n {diff}:\n {config_dificultad[diff]['meter']}:\n 0.1,0.1,0.1,0.1,0.1:\n")
            f.write(bloque)
            f.write("\n")

    # --- ESCRITURA DEL ARCHIVO .SSC MODERNO ---
    with open(output_ssc, "w", encoding="utf-8") as f:
        f.write(f"#VERSION:0.83;\n#TITLE:{song_title};\n#ARTIST:{artist_name};\n#MUSIC:{new_audio_name};\n#BANNER:{new_banner_name};\n")
        f.write(f"#VIDEO:{new_video_name};\n")
        f.write(f"#OFFSET:0.000;\n#BPMS:{bpms_string_line};\n#COMBOLINK:1;\n")

        # Se escribe la etiqueta de velocidades solo si fue la opción activa elegida
        if speeds_activos_y_excluyentes := (speeds_activo and not dinamico_activo and speeds_string_line):
            f.write(f"#SPEEDS:{speeds_string_line};\n")

        f.write(f"#BGCHANGES:{bg_changes_line};\n\n")
        
        for diff, bloque in mapa_pasos_por_dificultad.items():
            f.write(f"//dance-single - AI_DSP_Hybrid_Engine\n#NOTEDATA:;\n#CHARTNAME:;\n#STEPSTYPE:dance-single;\n")
            f.write(f"#DESCRIPTION:AI_DSP_Hybrid_Engine;\n#DIFFICULTY:{diff};\n#METER:{config_dificultad[diff]['meter']};\n")
            f.write(f"#RADARVALUES:0.1,0.1,0.1,0.1,0.1;\n#CREDIT:AI_Engine;\n#NOTES:\n")
            f.write(bloque)
            f.write("\n")

    # Modificamos el reporte de métricas:
    log_metricas_diff += f"⏱️ OFFSET DETECTADO/APLICADO: {val_offset:.3f} s\n"

    if custom_params.get("recalcular_dificultad", False):
        # Añadimos la huella digital al bloque de texto que va para la consola
        reporte_con_huella = (
            f"{log_metricas_diff}"
            f"🔑 HUELLA DIGITAL (SEED): {seed_actual}\n"
            f"💡 Guarda este número si deseas replicar exactamente este mismo patrón.\n"
            f"{'-'*45}\n"
        )
        log_metricas_diff = reporte_con_huella
    else:
        # En caso de que tengan el cálculo dinámico apagado, igual les mostramos la semilla
        log_metricas_diff = (
            f"🔑 HUELLA DIGITAL (SEED): {seed_actual}\n"
            f"└─ (Cálculo dinámico de dificultad desactivado)\n"
            f"{'-'*45}\n"
        )
            
    return bpm, duracion, log_metricas_diff

# =====================================================================
# 4. INTERFAZ GRÁFICA COMPATIBLE ADAPTATIVA CON SCROLL
# =====================================================================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class StepHybridUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("StepMania AI + DSP Dual Simfile Generator")
        self.geometry("480x720") # Ancho optimizado para distribución completamente vertical fija
        self.resizable(True, True)
        self.minsize(420, 550)
        self.audio_file_path = ""
        self.checkpoint_file_path = ""
        self.banner_file_path = ""
        self.video_file_path = ""
        self.minas_rms_activa = False

        self.label_titulo = ctk.CTkLabel(self, text="AI + DSP Chart Generator (Dual SM/SSC)", font=ctk.CTkFont(size=16, weight="bold"))
        self.label_titulo.pack(pady=10, fill="x")
        self.scroll_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.inicializar_componentes()

    def alternar_visibilidad_postprocesamiento(self):
        """ Activa o desactiva sliders avanzados según el estado del checkbox principal """
        estado = "normal" if self.checkbox_postprocesar.get() else "disabled"
        self.slider_temp.configure(state=estado)
        self.slider_max_hold.configure(state=estado)
        self.slider_holds_sim.configure(state=estado)
        self.slider_prob_minas.configure(state=estado)
        self.slider_max_minas.configure(state=estado)

    def inicializar_componentes(self):
        # Todo se empaqueta secuencialmente en un único contenedor lineal estable
        self.contenedor_vertical = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        self.contenedor_vertical.pack(fill="x", expand=True, padx=5)

        self.btn_audio = ctk.CTkButton(self.contenedor_vertical, text="1. Seleccionar Canción (.mp3, .wav)", fg_color="#34495e", command=self.buscar_audio)
        self.label_audio_path = ctk.CTkLabel(self.contenedor_vertical, text="Ningún archivo seleccionado", text_color="gray", wraplength=350)

        self.btn_checkpoint = ctk.CTkButton(self.contenedor_vertical, text="2. Seleccionar Checkpoint IA (.pt)", fg_color="#2c3e50", command=self.buscar_checkpoint)
        self.label_checkpoint_path = ctk.CTkLabel(self.contenedor_vertical, text="Ningún modelo cargado", text_color="gray", wraplength=350)
        #Verificando si existe el checkpoint por default
        self.checkpoint_path = os.path.join(checkpoint_dir, model_name)
        if os.path.exists(self.checkpoint_path):
            print(f" 📦 Cargando pesos desde checkpoint histórico: {self.checkpoint_path}")
            self.checkpoint_file_path = self.checkpoint_path
            self.label_checkpoint_path.configure(text=f"{self.checkpoint_path}", text_color="gray")

        
        self.label_name = ctk.CTkLabel(self.contenedor_vertical, text="Título de la Canción:", font=ctk.CTkFont(weight="bold"))
        self.entry_title = ctk.CTkEntry(self.contenedor_vertical, placeholder_text="Ej: Cybernetic Beats", width=340)

        self.label_artist_name = ctk.CTkLabel(self.contenedor_vertical, text="Nombre del artista:", font=ctk.CTkFont(weight="bold"))
        self.entry_artist_name = ctk.CTkEntry(self.contenedor_vertical, placeholder_text="Ej: AI", width=340)

        self.label_duracion = ctk.CTkLabel(self.contenedor_vertical, text="Duración Máxima (Segundos / 0=Full):", font=ctk.CTkFont(weight="bold"))
        self.entry_duracion = ctk.CTkEntry(self.contenedor_vertical, placeholder_text="Ej: 90", width=340)

        self.label_seed = ctk.CTkLabel(self.contenedor_vertical, text="Semilla de Generación (Vacío = Aleatorio):", font=ctk.CTkFont(weight="bold"))
        self.entry_seed = ctk.CTkEntry(self.contenedor_vertical, placeholder_text="Ej: 12345 o texto_libre", width=340)
        
        self.checkbox_rename = ctk.CTkCheckBox(self.contenedor_vertical, text="Renombrar archivos (nombre_cancion_(banner/video))") 

        self.label_bpm = ctk.CTkLabel(self.contenedor_vertical, text="Configuración de BPM: Auto (DSP)", font=ctk.CTkFont(weight="bold"))
        self.slider_bpm = ctk.CTkSlider(self.contenedor_vertical, from_=0, to=300, number_of_steps=240, command=self.actualizar_texto_bpm, width=340)
        self.slider_bpm.set(0)

        # --- BOTONES DE AJUSTE FINO ---
        self.frame_bpm_botones = ctk.CTkFrame(self.contenedor_vertical, fg_color="transparent")
        self.btn_bpm_menos = ctk.CTkButton(self.frame_bpm_botones, text="- 1", width=80, fg_color="#7f8c8d", hover_color="#95a5a6", command=self.decrementar_bpm)
        self.btn_bpm_menos.pack(side="left", padx=10)
        self.btn_bpm_mas = ctk.CTkButton(self.frame_bpm_botones, text="+ 1", width=80, fg_color="#7f8c8d", hover_color="#95a5a6", command=self.incrementar_bpm)
        self.btn_bpm_mas.pack(side="left", padx=10)
        
        self.checkbox_bpm = ctk.CTkCheckBox(self.contenedor_vertical, text="Doble BPM (x2)") 
        
        # Checkboxes configurados con el comando de exclusión mutua
        self.checkbox_bpm_dinamico = ctk.CTkCheckBox(
            self.contenedor_vertical, 
            text="Aplicar BPM Dinámico (Alteraciones)",
            command=self.gestionar_exclusividad_ritmo
        )
        
        self.checkbox_speeds_dinamico = ctk.CTkCheckBox(
            self.contenedor_vertical, 
            text="Adaptar Velocidad Visual (Scroll Speeds)",
            command=self.gestionar_exclusividad_ritmo
        )
        
        # --- SLIDERS DE CONTROL DE VELOCIDAD EXTREMA ---
        self.label_speed_min = ctk.CTkLabel(self.contenedor_vertical, text="Scroll Mínimo (Calma): 0.70x", font=ctk.CTkFont(weight="bold"))
        self.slider_speed_min = ctk.CTkSlider(self.contenedor_vertical, from_=0.25, to=1.0, number_of_steps=15, width=340, command=lambda v: self.label_speed_min.configure(text=f"Scroll Mínimo (Calma): {v:.2f}x"))
        self.slider_speed_min.set(0.70)

        self.label_speed_max = ctk.CTkLabel(self.contenedor_vertical, text="Scroll Máximo (Drop): 1.40x", font=ctk.CTkFont(weight="bold"))
        self.slider_speed_max = ctk.CTkSlider(self.contenedor_vertical, from_=1.0, to=3.0, number_of_steps=40, width=340, command=lambda v: self.label_speed_max.configure(text=f"Scroll Máximo (Drop): {v:.2f}x"))
        self.slider_speed_max.set(1.40)

        # --- SLIDER PARA DURACIÓN DE TRANSICIÓN ---
        self.label_speed_trans = ctk.CTkLabel(self.contenedor_vertical, text="Duración de Transición: 2.0 Beats (Suave)", font=ctk.CTkFont(weight="bold"))
        self.slider_speed_trans = ctk.CTkSlider(self.contenedor_vertical, from_=0.0, to=4.0, number_of_steps=16, width=340, command=lambda v: self.label_speed_trans.configure(text=f"Duración de Transición: {v:.1f} Beats" if v > 0 else "Duración de Transición: Inmediata (0.0)"))
        self.slider_speed_trans.set(2.0)

        # Checkbox principal iniciado por defecto en True (Habilitado)
        self.checkbox_postprocesar = ctk.CTkCheckBox(self.contenedor_vertical, text="Aplicar Posprocesamiento rítmico", command=self.alternar_visibilidad_postprocesamiento)
        self.checkbox_postprocesar.select()

        # --- CHECKBOX PARA RECALCULAR METER ---
        self.checkbox_recalcular_diff = ctk.CTkCheckBox(self.contenedor_vertical, text="Recalcular Dificultad Dinámicamente (NPS)", text_color="#3498db")
        self.checkbox_recalcular_diff.select() # Activado por defecto

        self.label_offset = ctk.CTkLabel(self.contenedor_vertical, text="Offset de Inicio: 0.000 s (Por defecto)", font=ctk.CTkFont(weight="bold"))
        self.slider_offset = ctk.CTkSlider(self.contenedor_vertical, from_=0.0, to=16.0, number_of_steps=400, width=340, command=self.actualizar_texto_offset)
        self.slider_offset.set(0.000)

        # --- BOTONES DE AJUSTE FINO ---
        self.frame_offset_botones = ctk.CTkFrame(self.contenedor_vertical, fg_color="transparent")
        self.btn_offset_menos = ctk.CTkButton(self.frame_offset_botones, text="- 0.04s", width=80, fg_color="#7f8c8d", hover_color="#95a5a6", command=self.decrementar_offset)
        self.btn_offset_menos.pack(side="left", padx=10)
        self.btn_offset_mas = ctk.CTkButton(self.frame_offset_botones, text="+ 0.04s", width=80, fg_color="#7f8c8d", hover_color="#95a5a6", command=self.incrementar_offset)
        self.btn_offset_mas.pack(side="left", padx=10)

        self.checkbox_offset_auto = ctk.CTkCheckBox(
            self.contenedor_vertical, 
            text="Detectar Offset Automáticamente (DSP Vol)",
            text_color="#1abc9c",
            command=self.gestionar_exclusividad_offset
        )
        self.checkbox_offset_auto.select()
        #self.checkbox_offset_auto.pack(pady=5, padx=20) # Asegúrate de añadirlo a componentes_ui más abajo si usas el bucle de empaquetado

        # === Extensión final de la canción ===
        self.label_extension = ctk.CTkLabel(self.contenedor_vertical, text="Extensión Final Estética: 0.0 s (Corte normal)", font=ctk.CTkFont(weight="bold"))
        self.slider_extension = ctk.CTkSlider(self.contenedor_vertical, from_=0.0, to=15.0, number_of_steps=30, width=340, command=self.actualizar_texto_extension)
        self.slider_extension.set(0.0)
        
        self.btn_banner = ctk.CTkButton(self.contenedor_vertical, text="Seleccionar Banner Graphic (Opcional)", fg_color="#16a085", command=self.buscar_banner)
        self.label_banner_path = ctk.CTkLabel(self.contenedor_vertical, text="Ningún banner seleccionado", text_color="gray", wraplength=350)
        self.btn_video = ctk.CTkButton(self.contenedor_vertical, text="Seleccionar Video de Fondo (Opcional)", fg_color="#8e44ad", command=self.buscar_video)
        self.label_video_path = ctk.CTkLabel(self.contenedor_vertical, text="Ningún video seleccionado", text_color="gray", wraplength=350)

        # Separador visual lógico
        self.label_seccion_adv = ctk.CTkLabel(self.contenedor_vertical, text="--- Parámetros del Motor de Pasos ---", font=ctk.CTkFont(size=13, weight="bold", slant="italic"), text_color="#95a5a6")

        self.label_temp = ctk.CTkLabel(self.contenedor_vertical, text="Temperatura IA (Caos): 1.30", font=ctk.CTkFont(weight="bold"))
        self.slider_temp = ctk.CTkSlider(self.contenedor_vertical, from_=0.5, to=1.5, number_of_steps=20, width=340, command=lambda v: self.label_temp.configure(text=f"Temperatura IA (Caos): {v:.2f}"))
        self.slider_temp.set(1.3)
        
        self.label_max_hold = ctk.CTkLabel(self.contenedor_vertical, text="Duración Máxima de Hold: 8 líneas", font=ctk.CTkFont(weight="bold"))
        self.slider_max_hold = ctk.CTkSlider(self.contenedor_vertical, from_=2, to=32, number_of_steps=30, width=340, command=lambda v: self.label_max_hold.configure(text=f"Duración Máxima de Hold: {int(v)} líneas"))
        self.slider_max_hold.set(8)
        
        self.label_holds_sim = ctk.CTkLabel(self.contenedor_vertical, text="Máximo de Holds simultáneos: 2", font=ctk.CTkFont(weight="bold"))
        self.slider_holds_sim = ctk.CTkSlider(self.contenedor_vertical, from_=1, to=4, number_of_steps=3, width=340, command=lambda v: self.label_holds_sim.configure(text=f"Máximo de Holds simultáneos: {int(v)}"))
        self.slider_holds_sim.set(2)
        
        self.label_prob_minas = ctk.CTkLabel(self.contenedor_vertical, text="Probabilidad de Minas por compás: 35%", font=ctk.CTkFont(weight="bold"))
        self.slider_prob_minas = ctk.CTkSlider(self.contenedor_vertical, from_=0, to=100, number_of_steps=100, width=340, command=lambda v: self.label_prob_minas.configure(text=f"Probabilidad de Minas por compás: {int(v)}%"))
        self.slider_prob_minas.set(35)
        
        self.label_max_minas = ctk.CTkLabel(self.contenedor_vertical, text="Máximo Minas por Compás: 3", font=ctk.CTkFont(weight="bold"))
        self.slider_max_minas = ctk.CTkSlider(self.contenedor_vertical, from_=0, to=4, number_of_steps=4, width=340, command=lambda v: self.label_max_minas.configure(text=f"Máximo Minas por Compás: {int(v)}"))
        self.slider_max_minas.set(3)

        # --- CONTROLES ADICIONALES PARA NUEVOS TIPOS DE NOTAS ---
        self.label_prob_fakes = ctk.CTkLabel(self.contenedor_vertical, text="Probabilidad de Fakes por compás: 25%", font=ctk.CTkFont(weight="bold"))
        self.slider_prob_fakes = ctk.CTkSlider(self.contenedor_vertical, from_=0, to=100, number_of_steps=100, width=340, command=lambda v: self.label_prob_fakes.configure(text=f"Probabilidad de Fakes por compás: {int(v)}%"))
        self.slider_prob_fakes.set(25)

        self.label_max_fakes = ctk.CTkLabel(self.contenedor_vertical, text="Máximo Fakes por Compás: 0", font=ctk.CTkFont(weight="bold"))
        self.slider_max_fakes = ctk.CTkSlider(self.contenedor_vertical, from_=0, to=4, number_of_steps=4, width=340, command=lambda v: self.label_max_fakes.configure(text=f"Máximo Fakes por Compás: {int(v)}"))
        self.slider_max_fakes.set(0)

        self.label_prob_lifts = ctk.CTkLabel(self.contenedor_vertical, text="Probabilidad de Lifts por compás: 25%", font=ctk.CTkFont(weight="bold"))
        self.slider_prob_lifts = ctk.CTkSlider(self.contenedor_vertical, from_=0, to=100, number_of_steps=100, width=340, command=lambda v: self.label_prob_lifts.configure(text=f"Probabilidad de Lifts por compás: {int(v)}%"))
        self.slider_prob_lifts.set(25)

        self.label_max_lifts = ctk.CTkLabel(self.contenedor_vertical, text="Máximo Lifts por Compás: 0", font=ctk.CTkFont(weight="bold"))
        self.slider_max_lifts = ctk.CTkSlider(self.contenedor_vertical, from_=0, to=4, number_of_steps=4, width=340, command=lambda v: self.label_max_lifts.configure(text=f"Máximo Lifts por Compás: {int(v)}"))
        self.slider_max_lifts.set(0)

        self.label_prob_potions = ctk.CTkLabel(self.contenedor_vertical, text="Probabilidad de Potions por compás: 15%", font=ctk.CTkFont(weight="bold"))
        self.slider_prob_potions = ctk.CTkSlider(self.contenedor_vertical, from_=0, to=100, number_of_steps=100, width=340, command=lambda v: self.label_prob_potions.configure(text=f"Probabilidad de Potions por compás: {int(v)}%"))
        self.slider_prob_potions.set(15)

        self.label_max_potions = ctk.CTkLabel(self.contenedor_vertical, text="Máximo Potions por Compás: 0", font=ctk.CTkFont(weight="bold"))
        self.slider_max_potions = ctk.CTkSlider(self.contenedor_vertical, from_=0, to=4, number_of_steps=4, width=340, command=lambda v: self.label_max_potions.configure(text=f"Máximo Potions por Compás: {int(v)}"))
        self.slider_max_potions.set(0)

        self.label_prob_shields = ctk.CTkLabel(self.contenedor_vertical, text="Probabilidad de Shields por compás: 15%", font=ctk.CTkFont(weight="bold"))
        self.slider_prob_shields = ctk.CTkSlider(self.contenedor_vertical, from_=0, to=100, number_of_steps=100, width=340, command=lambda v: self.label_prob_shields.configure(text=f"Probabilidad de Shields por compás: {int(v)}%"))
        self.slider_prob_shields.set(15)

        self.label_max_shields = ctk.CTkLabel(self.contenedor_vertical, text="Máximo Shields por Compás: 0", font=ctk.CTkFont(weight="bold"))
        self.slider_max_shields = ctk.CTkSlider(self.contenedor_vertical, from_=0, to=4, number_of_steps=4, width=340, command=lambda v: self.label_max_shields.configure(text=f"Máximo Shields por Compás: {int(v)}"))
        self.slider_max_shields.set(0)

        self.label_prob_rayos = ctk.CTkLabel(self.contenedor_vertical, text="Probabilidad de Rayos por compás: 15%", font=ctk.CTkFont(weight="bold"))
        self.slider_prob_rayos = ctk.CTkSlider(self.contenedor_vertical, from_=0, to=100, number_of_steps=100, width=340, command=lambda v: self.label_prob_rayos.configure(text=f"Probabilidad de Rayos por compás: {int(v)}%"))
        self.slider_prob_rayos.set(15)

        self.label_max_rayos = ctk.CTkLabel(self.contenedor_vertical, text="Máximo Rayos por Compás: 0", font=ctk.CTkFont(weight="bold"))
        self.slider_max_rayos = ctk.CTkSlider(self.contenedor_vertical, from_=0, to=4, number_of_steps=4, width=340, command=lambda v: self.label_max_rayos.configure(text=f"Máximo Rayos por Compás: {int(v)}"))
        self.slider_max_rayos.set(0)

        self.label_prob_hiddens = ctk.CTkLabel(self.contenedor_vertical, text="Probabilidad de Hiddens por compás: 20%", font=ctk.CTkFont(weight="bold"))
        self.slider_prob_hiddens = ctk.CTkSlider(self.contenedor_vertical, from_=0, to=100, number_of_steps=100, width=340, command=lambda v: self.label_prob_hiddens.configure(text=f"Probabilidad de Hiddens por compás: {int(v)}%"))
        self.slider_prob_hiddens.set(20)

        self.label_max_hiddens = ctk.CTkLabel(self.contenedor_vertical, text="Máximo Hiddens por Compás: 0", font=ctk.CTkFont(weight="bold"))
        self.slider_max_hiddens = ctk.CTkSlider(self.contenedor_vertical, from_=0, to=4, number_of_steps=4, width=340, command=lambda v: self.label_max_hiddens.configure(text=f"Máximo Hiddens por Compás: {int(v)}"))
        self.slider_max_hiddens.set(0)

        self.checkbox_efectos_rms = ctk.CTkCheckBox(
            self.contenedor_vertical, 
            text="Potenciar Efectos y Trampas en Drops (Análisis RMS)",
            text_color="#e67e22"
        )
        self.checkbox_efectos_rms.select()

        self.checkbox_secciones_saltos = ctk.CTkCheckBox(
            self.contenedor_vertical, 
            text="Generar Secciones de Saltos (Filtro RMS)",
            text_color="#9b59b6"
        )

        self.label_rms_min = ctk.CTkLabel(self.contenedor_vertical, text="Sensibilidad RMS Mínimo (Calma): 0.50", font=ctk.CTkFont(weight="bold"))
        self.slider_rms_min = ctk.CTkSlider(self.contenedor_vertical, from_=0.1, to=1.0, number_of_steps=18, width=340, command=lambda v: self.label_rms_min.configure(text=f"Sensibilidad RMS Mínimo (Calma): {v:.2f}"))
        self.slider_rms_min.set(0.5)
        
        self.label_rms_max = ctk.CTkLabel(self.contenedor_vertical, text="Sensibilidad RMS Máximo (Drop): 1.50", font=ctk.CTkFont(weight="bold"))
        self.slider_rms_max = ctk.CTkSlider(self.contenedor_vertical, from_=1.0, to=2.5, number_of_steps=30, width=340, command=lambda v: self.label_rms_max.configure(text=f"Sensibilidad RMS Máximo (Drop): {v:.2f}"))
        self.slider_rms_max.set(1.5)
        
        self.label_nivel_texto = ctk.CTkLabel(self.contenedor_vertical, text=f"Dificultad Techo del Pack: Nivel 16/{MAX_LEVEL}", font=ctk.CTkFont(size=13, weight="bold"), text_color="#3498db")
        self.slider_level = ctk.CTkSlider(self.contenedor_vertical, from_=1, to=MAX_LEVEL, number_of_steps=MAX_LEVEL-1, width=340, command=self.actualizar_valores_interfaz)
        self.slider_level.set(16)

        self.btn_resetear = ctk.CTkButton(self.contenedor_vertical, text="Restablecer Parámetros", fg_color="#c0392b", hover_color="#962d22", command=self.restablecer_valores)
        self.btn_generar = ctk.CTkButton(self.contenedor_vertical, text="¡Procesar y Exportar Dual Pack! 🚀", fg_color="#2ecc71", hover_color="#27ae60", height=45, font=ctk.CTkFont(size=14, weight="bold"), command=self.iniciar_generacion)
        self.label_status = ctk.CTkLabel(self.contenedor_vertical, text="Estado: Esperando archivos mínimos...", font=ctk.CTkFont(size=12, weight="bold"), text_color="gray")

        # --- PANEL CONSOLA DE MÉTRICAS REALES ---
        self.label_consola = ctk.CTkLabel(self.contenedor_vertical, text="🖥️ Monitor de Densidad en Tiempo Real", font=ctk.CTkFont(size=12, weight="bold"))
        self.txt_consola = ctk.CTkTextbox(self.contenedor_vertical, height=140, width=340, font=ctk.CTkFont(family="Courier", size=11), fg_color="#1e272e", text_color="#2ecc71")
        self.txt_consola.insert("0.0", "Esperando ejecución para calcular NPS...\n")
        self.txt_consola.configure(state="disabled")

        # Empaquetado lineal descendente y ordenado para scroll seguro
        componentes_ui = [
            self.btn_audio, self.label_audio_path, self.btn_checkpoint, self.label_checkpoint_path,
            self.label_name, self.entry_title, self.checkbox_rename, self.label_artist_name, self.entry_artist_name,
            self.label_duracion, self.entry_duracion, 
            self.label_bpm, self.slider_bpm, self.frame_bpm_botones, self.checkbox_bpm, self.checkbox_bpm_dinamico, 
            self.checkbox_speeds_dinamico,
            self.label_speed_min, self.slider_speed_min,  
            self.label_speed_max, self.slider_speed_max,  
            self.label_speed_trans, self.slider_speed_trans,
            self.checkbox_postprocesar, self.checkbox_recalcular_diff, 
            self.label_offset, self.slider_offset, self.frame_offset_botones, self.checkbox_offset_auto, 
            self.label_extension, self.slider_extension,
            self.btn_banner, self.label_banner_path,
            self.btn_video, self.label_video_path, self.label_seccion_adv,
            self.label_temp, self.slider_temp, self.label_max_hold, self.slider_max_hold,
            self.label_holds_sim, self.slider_holds_sim, 
            self.label_prob_minas, self.slider_prob_minas, self.label_max_minas, self.slider_max_minas,
            self.label_prob_fakes, self.slider_prob_fakes, self.label_max_fakes, self.slider_max_fakes,
            self.label_prob_lifts, self.slider_prob_lifts, self.label_max_lifts, self.slider_max_lifts,
            self.label_prob_potions, self.slider_prob_potions, self.label_max_potions, self.slider_max_potions,
            self.label_prob_shields, self.slider_prob_shields, self.label_max_shields, self.slider_max_shields,
            self.label_prob_rayos, self.slider_prob_rayos, self.label_max_rayos, self.slider_max_rayos,
            self.label_prob_hiddens, self.slider_prob_hiddens, self.label_max_hiddens, self.slider_max_hiddens,
            self.checkbox_efectos_rms,
            self.checkbox_secciones_saltos,
            self.label_rms_min, self.slider_rms_min, self.label_rms_max, self.slider_rms_max,
            self.label_nivel_texto, self.slider_level, self.label_seed, self.entry_seed, self.btn_resetear, self.btn_generar, self.label_status,
            self.label_consola, self.txt_consola
        ]
        
        for widget in componentes_ui:
            widget.pack(pady=5, fill="x" if "Button" in type(widget).__name__ else None, padx=20)

        # --- NUEVO MÉTODO AGREGADO ---
    def restablecer_valores(self):
        """ Devuelve todos los sliders avanzados a sus valores nativos por defecto """
        # --- LIMPIEZA DE RUTAS Y CAMPOS DE ARCHIVOS ---
        self.audio_file_path = ""
        #self.checkpoint_file_path = ""
        self.banner_file_path = ""
        self.video_file_path = ""
        
        self.label_audio_path.configure(text="Ningún archivo seleccionado", text_color="gray")
        #self.label_checkpoint_path.configure(text="Ningún modelo cargado", text_color="gray")
        self.label_banner_path.configure(text="Ningún banner seleccionado", text_color="gray")
        self.label_video_path.configure(text="Ningún video seleccionado", text_color="gray")
        
        self.entry_title.delete(0, "end")
        self.entry_artist_name.delete(0, "end")
        self.entry_duracion.delete(0, "end")
        self.entry_seed.delete(0, "end")

        # Resetear Sincronización Básica
        self.checkbox_rename.deselect()

        self.slider_bpm.set(0)
        self.label_bpm.configure(text="Configuración de BPM: Auto (Detección DSP)")

        self.checkbox_bpm.deselect() 

        self.checkbox_bpm_dinamico.deselect() 
        self.checkbox_bpm_dinamico.configure(state="normal")
        self.checkbox_speeds_dinamico.deselect()
        self.checkbox_speeds_dinamico.configure(state="normal")

        # Reset de los sliders de velocidad
        self.slider_speed_min.set(0.70)
        self.slider_speed_min.configure(state="normal")
        self.label_speed_min.configure(text="Scroll Mínimo (Calma): 0.70x")
        self.slider_speed_max.set(1.40)
        self.slider_speed_max.configure(state="normal")
        self.label_speed_max.configure(text="Scroll Máximo (Drop): 1.40x")

        self.slider_speed_trans.set(2.0)
        self.slider_speed_trans.configure(state="normal")
        self.label_speed_trans.configure(text="Duración de Transición: 2.0 Beats (Suave)")

        self.checkbox_postprocesar.select()
        self.checkbox_recalcular_diff.select() # Reset a activado
        
        self.slider_offset.set(0.000)
        self.label_offset.configure(text="Offset de Inicio: 0.000 s (Por defecto)")

        self.checkbox_offset_auto.select()
        self.slider_offset.configure(state="disabled")

        self.slider_extension.set(0.000)
        self.label_extension.configure(text="Extensión Final Estética: 0.0 s (Corte normal)")
        
        # Resetear Parámetros Avanzados
        self.slider_temp.set(1.3)
        self.label_temp.configure(text="Temperatura IA (Caos): 1.30")
        
        self.slider_max_hold.set(8)
        self.label_max_hold.configure(text="Duración Máxima de Hold: 8 líneas")
        
        self.slider_holds_sim.set(2)
        self.label_holds_sim.configure(text="Máximo de Holds simultáneos: 2")
        
        self.slider_prob_minas.set(35)
        self.label_prob_minas.configure(text="Probabilidad de Minas por compás: 35%")
        
        self.slider_max_minas.set(3)
        self.label_max_minas.configure(text="Máximo Minas por Compás: 3")

        self.checkbox_secciones_saltos.deselect()
        
        self.slider_rms_min.set(0.5)
        self.label_rms_min.configure(text="Sensibilidad RMS Mínimo (Calma): 0.50")
        
        self.slider_rms_max.set(1.5)
        self.label_rms_max.configure(text="Sensibilidad RMS Máximo (Drop): 1.50")
        
        # Resetear Techo de Dificultad
        self.slider_level.set(16)
        self.actualizar_valores_interfaz()
        
        self.label_status.configure(text="Estado: Parámetros restablecidos correctamente.", text_color="gray")

        self.txt_consola.configure(state="normal")
        self.txt_consola.delete("0.0", "end")
        self.txt_consola.insert("0.0", "Esperando ejecución para calcular NPS...\n")
        self.txt_consola.configure(state="disabled")

        self.slider_prob_fakes.set(25)
        self.label_prob_fakes.configure(text="Probabilidad de Fakes por compás: 25%")
        self.slider_max_fakes.set(0)
        self.label_max_fakes.configure(text="Máximo Fakes por Compás: 0")

        self.slider_prob_lifts.set(25)
        self.label_prob_lifts.configure(text="Probabilidad de Lifts por compás: 25%")
        self.slider_max_lifts.set(0)
        self.label_max_lifts.configure(text="Máximo Lifts por Compás: 0")
        
        self.slider_prob_potions.set(15)
        self.label_prob_potions.configure(text="Probabilidad de Potions por compás: 15%")
        self.slider_max_potions.set(0)
        self.label_max_potions.configure(text="Máximo Potions por Compás: 0")

        self.slider_prob_shields.set(15)
        self.label_prob_shields.configure(text="Probabilidad de Shields por compás: 15%")
        self.slider_max_shields.set(0)
        self.label_max_shields.configure(text="Máximo Shields por Compás: 0")

        self.slider_prob_rayos.set(15)
        self.label_prob_rayos.configure(text="Probabilidad de Rayos por compás: 15%")
        self.slider_max_rayos.set(0)
        self.label_max_rayos.configure(text="Máximo Rayos por Compás: 0")

        self.slider_prob_hiddens.set(20)
        self.label_prob_hiddens.configure(text="Probabilidad de Hiddens por compás: 20%")
        self.slider_max_hiddens.set(0)
        self.label_max_hiddens.configure(text="Máximo Hiddens por Compás: 0")

        self.checkbox_efectos_rms.select()


    def actualizar_texto_bpm(self, valor):
        bpm_int = int(valor)
        if bpm_int == 0:
            self.label_bpm.configure(text="Configuración de BPM: Auto (Detección DSP)")
        else:
            if bpm_int < 60:
                bpm_int = 60
                self.slider_bpm.set(60)
            self.label_bpm.configure(text=f"Configuración de BPM: {bpm_int} BPM (Manual)")
    
    def gestionar_exclusividad_ritmo(self):
        if self.checkbox_bpm_dinamico.get():
            self.checkbox_speeds_dinamico.configure(state="disabled")
            self.slider_speed_min.configure(state="disabled")
            self.slider_speed_max.configure(state="disabled")
            self.slider_speed_trans.configure(state="disabled") # Bloqueo
        elif self.checkbox_speeds_dinamico.get():
            self.checkbox_bpm_dinamico.configure(state="disabled")
            self.slider_speed_min.configure(state="normal")
            self.slider_speed_max.configure(state="normal")
            self.slider_speed_trans.configure(state="normal") # Desbloqueo
        else:
            self.checkbox_bpm_dinamico.configure(state="normal")
            self.checkbox_speeds_dinamico.configure(state="normal")
            self.slider_speed_min.configure(state="normal")
            self.slider_speed_max.configure(state="normal")

    def actualizar_texto_offset(self, valor):
        self.label_offset.configure(text=f"Offset de Inicio: {valor:.3f} s")

    def incrementar_offset(self):
        # Incrementa en un paso del slider (16.0 / 400 = 0.04s)
        nuevo_valor = min(16.0, self.slider_offset.get() + 0.04)
        self.slider_offset.set(nuevo_valor)
        self.actualizar_texto_offset(nuevo_valor)

    def decrementar_offset(self):
        # Decrementa en un paso del slider (16.0 / 400 = 0.04s)
        nuevo_valor = max(0.0, self.slider_offset.get() - 0.04)
        self.slider_offset.set(nuevo_valor)
        self.actualizar_texto_offset(nuevo_valor)
    
    def gestionar_exclusividad_offset(self):
        """Desactiva los controles manuales de offset si el modo automático está encendido."""
        if self.checkbox_offset_auto.get():
            self.slider_offset.configure(state="disabled")
            self.btn_offset_menos.configure(state="disabled")
            self.btn_offset_mas.configure(state="disabled")
            self.label_offset.configure(text="Offset de Inicio: [Automático Activo]")
        else:
            self.slider_offset.configure(state="normal")
            self.btn_offset_menos.configure(state="normal")
            self.btn_offset_mas.configure(state="normal")
            self.actualizar_texto_offset(self.slider_offset.get())

    def incrementar_bpm(self):
        # Incrementa en un paso del slider (60 -> 300)
        nuevo_valor = min(300, self.slider_bpm.get() + 1)
        if nuevo_valor < 60:
            nuevo_valor = 60
        self.slider_bpm.set(nuevo_valor)
        self.actualizar_texto_bpm(nuevo_valor)

    def decrementar_bpm(self):
        # Decrementa en un paso del slider (0 <- 300)
        nuevo_valor = max(59, self.slider_bpm.get() - 1)
        if nuevo_valor < 60:
            nuevo_valor = 0
        self.slider_bpm.set(nuevo_valor)
        self.actualizar_texto_bpm(nuevo_valor)
    
    def actualizar_texto_extension(self, valor):
        self.label_extension.configure(text=f"Extensión Final Estética: {valor:.1f} s")

    def actualizar_valores_interfaz(self, *args):
        level = int(self.slider_level.get())
        e = max(1, int(level * 0.25))
        m = max(2, int(level * 0.50))
        h = max(3, int(level * 0.75))
        self.label_nivel_texto.configure(text=f"Dificultad Techo: Nivel {level}\nEscala: [Easy {e} | Med {m} | Hard {h} | Chal {level}]")

    def buscar_audio(self):
        file_path = filedialog.askopenfilename(filetypes=[("Archivos de Audio", "*.mp3 *.wav *.ogg *.flac")])
        if file_path:
            self.audio_file_path = file_path
            self.label_audio_path.configure(text=os.path.basename(file_path), text_color="#1abc9c")
            self.entry_title.delete(0, "end")
            self.entry_title.insert(0, os.path.splitext(os.path.basename(file_path))[0])

    def buscar_checkpoint(self):
        file_path = filedialog.askopenfilename(filetypes=[("PyTorch Checkpoints", "*.pt")])
        if file_path:
            self.checkpoint_file_path = file_path
            self.label_checkpoint_path.configure(text=os.path.basename(file_path), text_color="#1abc9c")

    def buscar_banner(self):
        file_path = filedialog.askopenfilename(filetypes=[("Gráficos de Banner", "*.png *.jpg *.jpeg *.bpm")])
        if file_path:
            self.banner_file_path = file_path
            self.label_banner_path.configure(text=os.path.basename(file_path), text_color="#1abc9c")

    def buscar_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Archivos de Video", "*.mp4 *.avi *.mkv *.flv *.mpg")])
        if file_path:
            self.video_file_path = file_path
            self.label_video_path.configure(text=os.path.basename(file_path), text_color="#1abc9c")

    def iniciar_generacion(self):
        if not self.audio_file_path or not self.checkpoint_file_path:
            messagebox.showerror("Error", "Debes cargar obligatoriamente el audio y el checkpoint (.pt) de la IA.")
            return
        titulo = self.entry_title.get().strip()
        if not titulo:
            messagebox.showerror("Error", "El título del simfile no puede estar vacío.")
            return

        duracion_texto = self.entry_duracion.get().strip()
        duracion_manual = 0.0
        if duracion_texto:
            try:
                duracion_manual = float(duracion_texto)
                if duracion_manual < 0: raise ValueError
            except ValueError:
                messagebox.showerror("Error", "La duración debe ser un número válido.")
                return
        
        seed_raw = self.entry_seed.get().strip()
        if not seed_raw:
            # Si está vacío, generamos una semilla aleatoria alejada de 0
            seed_final = random.randint(100000, 999999)
        else:
            try:
                # Si es un número puro, lo usamos directamente
                seed_final = int(seed_raw)
            except ValueError:
                # Si es texto (ej: "mi_cancion_épica"), lo convertimos a un entero válido y estable
                seed_final = abs(hash(seed_raw)) % (10**8)

        params_usuario = {
            "renombrar_archivos": bool(self.checkbox_rename.get()),
            "bpm_manual": int(self.slider_bpm.get()),
            "double_bpm": bool(self.checkbox_bpm.get()),
            "aplicar_bpm_dinamico": bool(self.checkbox_bpm_dinamico.get()), # Mapeo del parámetro dinámico
            "aplicar_speeds_dinamicos": bool(self.checkbox_speeds_dinamico.get()),
            "speed_min_custom": float(self.slider_speed_min.get()),  
            "speed_max_custom": float(self.slider_speed_max.get()),  
            "speed_trans_custom": float(self.slider_speed_trans.get()),
            "aplicar_postprocesamiento": bool(self.checkbox_postprocesar.get()),
            "recalcular_dificultad": bool(self.checkbox_recalcular_diff.get()),
            "offset_automatico": bool(self.checkbox_offset_auto.get()),
            "offset_manual": float(self.slider_offset.get()),
            "temperatura": float(self.slider_temp.get()),
            "max_hold": int(self.slider_max_hold.get()),
            "holds_simultaneos": int(self.slider_holds_sim.get()),
            "activar_secciones_saltos": bool(self.checkbox_secciones_saltos.get()),
            "ratio_min_energia": float(self.slider_rms_min.get()),
            "ratio_max_energia": float(self.slider_rms_max.get()),
            "extension_final": float(self.slider_extension.get()),
            # Extracción de valores de la UI para los efectos
            "probabilidad_minas": float(self.slider_prob_minas.get()) / 100.0,
            "max_minas_compas": int(self.slider_max_minas.get()),
            "probabilidad_fakes": float(self.slider_prob_fakes.get()) / 100.0,
            "max_fakes_compas": int(self.slider_max_fakes.get()),
            "probabilidad_lifts": float(self.slider_prob_lifts.get()) / 100.0,
            "max_lifts_compas": int(self.slider_max_lifts.get()),
            "probabilidad_potions": float(self.slider_prob_potions.get()) / 100.0,
            "max_potions_compas": int(self.slider_max_potions.get()),
            "probabilidad_shields": float(self.slider_prob_shields.get()) / 100.0,
            "max_shields_compas": int(self.slider_max_shields.get()),
            "probabilidad_rayos": float(self.slider_prob_rayos.get()) / 100.0,
            "max_rayos_compas": int(self.slider_max_rayos.get()),
            "probabilidad_hiddens": float(self.slider_prob_hiddens.get()) / 100.0,
            "max_hiddens_compas": int(self.slider_max_hiddens.get()),
            "efectos_por_rms": bool(self.checkbox_efectos_rms.get()),
            "seed_value": seed_final
        }

        self.btn_generar.configure(state="disabled", text="Ejecutando Inferencia Híbrida ⚡...")
        self.label_status.configure(text="Estado: Procesando matrices y DSP...", text_color="#f1c40f")
        
        threading.Thread(target=self.ejecutar_proceso, args=(titulo, duracion_manual, params_usuario)).start()

    def actualizar_consola_gui(self, texto):
        self.txt_consola.configure(state="normal")
        self.txt_consola.delete("0.0", "end")
        self.txt_consola.insert("0.0", texto)
        self.txt_consola.configure(state="disabled")

    def ejecutar_proceso(self, titulo, duracion_manual, params_usuario):
        max_nivel = int(self.slider_level.get())
        carpeta_salida = os.path.dirname(self.audio_file_path)
        try:
            # Modificación de la llamada para recibir el tercer parámetro del core
            bpm, duracion, reporte_nps = generar_simfiles_hibridos(
                audio_path=self.audio_file_path,
                checkpoint_path=self.checkpoint_file_path,
                song_title=titulo,
                max_level_chosen=max_nivel,
                carpeta_salida=carpeta_salida,
                artist_name=self.entry_artist_name.get().strip(),
                banner_path=self.banner_file_path,
                video_path=self.video_file_path,
                duracion_limite=duracion_manual,
                custom_params=params_usuario
            )
            
            # Enviar el reporte de texto a la GUI de manera segura
            if params_usuario.get("recalcular_dificultad", False):
                self.after(0, self.actualizar_consola_gui, reporte_nps)
            else:
                self.after(0, self.actualizar_consola_gui, "Recálculo desactivado. Se usaron niveles base del GUI.\n")

            self.label_status.configure(text="¡ÉXITO: Archivos creados! ✅", text_color="#2ecc71")
            messagebox.showinfo("Proceso Completado", f"Pack Híbrido Creado con Éxito.\n\nArchivos .sm y .ssc listos.\nBPM: {bpm:.2f} | Duración: {duracion:.1f}s")

        except Exception as e:
            self.label_status.configure(text="Error Crítico ❌", text_color="#e74c3c")
            messagebox.showerror("Error de Inferencia", str(e))
        finally:
            self.btn_generar.configure(state="normal", text="¡Procesar y Exportar Dual Pack! 🚀")
        
        #Limpieza de ciertos campos
        self.audio_file_path = ""
        self.banner_file_path = ""
        self.video_file_path = ""
        self.label_audio_path.configure(text="Ningún archivo seleccionado", text_color="gray")
        self.label_banner_path.configure(text="Ningún banner seleccionado", text_color="gray")
        self.label_video_path.configure(text="Ningún video seleccionado", text_color="gray")

if __name__ == "__main__":
    app = StepHybridUI()
    app.mainloop()
