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
# --- IMPORTACIONES PARA EL MOTOR GRÁFICO INTERACTIVO ---
import time
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
# CORE DE IA
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
    def recalcular_meter_real(secuencia_pasos, duracion_segundos, dificultad_tag, max_level_chosen, 
                              max_notas_compas=12, extension_final=0.0, custom_params=None, lineas_por_compas=12, bpm=120.0,
                              factor_escala_muestreo=1.0):
        """
        Calcula el METER real basado en el PICO MÁXIMO de densidad (NPS Local) mediante ventanas móviles de 8 compases.
        CORREGIDO: Escala lineal y exponencialmente el NPS resultante basándose en la capa de muestreo extraída.
        """
        duracion_efectiva = max(1.0, duracion_segundos - extension_final)
        if not secuencia_pasos or duracion_efectiva <= 0:
            return 1, ""
            
        total_impactos_global = 0
        total_modificadores_global = 0
        
        for paso in secuencia_pasos:
            total_impactos_global += sum(1 for char in paso if char in ['1', '2', '4'])
            total_modificadores_global += sum(1 for char in paso if char in ['M', 'F', 'L', 'S', 'H', 'D', 'P'])
            
        nps_promedio_global = total_impactos_global / duracion_efectiva

        # ---------------------------------------------------------------------
        # ESCANEO DE VENTANAS LOCALES (Picos de Densidad Rítmica)
        # ---------------------------------------------------------------------
        segundos_por_compas = (60.0 / bpm) * 4.0
        lineas_por_bloque = lineas_por_compas * 8
        segundos_por_bloque = segundos_por_compas * 8
        
        nps_maximo_local = 0.0
        total_lineas = len(secuencia_pasos)
        
        for i in range(0, total_lineas, lineas_por_compas):
            fin_bloque = min(i + lineas_por_bloque, total_lineas)
            sub_secuencia = secuencia_pasos[i:fin_bloque]
            
            if not sub_secuencia:
                continue
                
            impactos_locales = sum(sum(1 for char in paso if char in ['1', '2', '4']) for paso in sub_secuencia)
            proporcion_bloque = len(sub_secuencia) / lineas_por_bloque
            tiempo_bloque_real = segundos_por_bloque * proporcion_bloque
            
            if tiempo_bloque_real > 0:
                nps_local = impactos_locales / tiempo_bloque_real
                if nps_local > nps_maximo_local:
                    nps_maximo_local = nps_local

        if nps_maximo_local == 0.0:
            nps_maximo_local = nps_promedio_global

        # Multiplicador de estrés coexistente
        factor_estres_dinamico = 1.0
        if custom_params:
            if custom_params.get("aplicar_bpm_dinamico", False):
                factor_estres_dinamico += 0.12
                
            if custom_params.get("aplicar_speeds_dinamicos", False):
                speed_max = custom_params.get("speed_max_custom", 1.40)
                if speed_max > 1.20:
                    factor_estres_dinamico += (speed_max - 1.20) * 0.45

        # Ponderación base de NPS
        nps_ponderado = (nps_maximo_local * 0.70) + (nps_promedio_global * 0.30)
        
        # --- CONTROL ADAPTATIVO DEL PRECIO DE DIFICULTAD (MUESTREO) ---
        # Multiplicamos el NPS por el factor de muestreo. Al usar la potencia (1.2)
        # los mapas muy recortados bajan más agresivamente su nivel (Garantiza escalabilidad jugable)
        nps_ponderado = nps_ponderado * (factor_escala_muestreo ** 1.2)

        nps_ajustado_base = nps_ponderado * factor_estres_dinamico

        PROMEDIO_BASE = 12.0
        factor_exponencial_notas = (max_notas_compas / PROMEDIO_BASE) ** 2
        nps_ajustado = nps_ajustado_base * factor_exponencial_notas

        # Techos de dificultad dinámicos mitigados proporcionalmente según la capa de muestreo
        techo_easy = max(1, int((max_level_chosen * 0.25) * factor_escala_muestreo))
        techo_med  = max(3, int((max_level_chosen * 0.50) * factor_escala_muestreo))
        techo_hard = max(6, int((max_level_chosen * 0.75) * factor_escala_muestreo))
        techo_chal = max(12, int(max_level_chosen * factor_escala_muestreo))

        factor_peligro_trampas = min(2.5, (total_modificadores_global / (total_impactos_global + 1)) * 6.0)

        # Asignación final con límites dinámicos escalados
        if dificultad_tag == "Easy":
            meter_estimado = int(nps_ajustado * 2.3) + 1
            meter_real = max(1, min(meter_estimado, techo_easy))
        elif dificultad_tag == "Medium":
            meter_estimado = int(nps_ajustado * 2.8) + 2 + int(factor_peligro_trampas)
            meter_real = max(techo_easy + 1, min(meter_estimado, techo_med))
        elif dificultad_tag == "Hard":
            meter_estimado = int((nps_ajustado * 3.3) + 3 + factor_peligro_trampas)
            meter_real = max(techo_med + 1, min(meter_estimado, techo_hard))
        else: # Challenge
            meter_estimado = int((nps_ajustado * 3.9) + 4 + (factor_peligro_trampas * 1.5))
            meter_real = max(techo_hard + 1, min(meter_estimado, techo_chal))

        bloques_barra = "█" * min(30, meter_real)
        espacios_barra = "░" * max(0, (30 - meter_real))
        
        reporte = (
            f"📊 [{dificultad_tag.upper()}] (Capa {int(factor_escala_muestreo*100)}%) NPS Glob: {nps_promedio_global*factor_escala_muestreo:.2f}\n"
            f"🎯 METER DINÁMICO ESCALADO: Nivel {meter_real} (Techo Máx Capa: {techo_chal})\n"
            f"└─ [{bloques_barra}{espacios_barra}]\n"
            f"{'-'*45}\n"
        )

        return meter_real, reporte


# =====================================================================
# MOTOR DSP ESPECTRAL (OPTIMIZADO CON RMS ASIMÉTRICOS)
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
        
        # --- EXTRACCIÓN DE RMS INDEPENDIENTES Y ASIMÉTRICOS ---
        # 1. RMS para BPM Dinámico: Ventana macro (1024) para suavizar micro-picos y evaluar tendencias estables
        rms_bpm = librosa.feature.rms(y=y, hop_length=1024).flatten()
        rms_medio_bpm = float(rms_bpm.mean()) if len(rms_bpm) > 0 else 1.0
        
        # 2. RMS para Scroll Speeds: Ventana estándar (512) para reaccionar fluidamente a cambios de intensidad visual
        rms_speed = librosa.feature.rms(y=y, hop_length=512).flatten()
        rms_medio_speed = float(rms_speed.mean()) if len(rms_speed) > 0 else 1.0
        
        # 3. RMS para Saltos y Efectos: Ventana micro (256) de alta resolución para capturar transitorios y golpes secos (Drops)
        rms_saltos = librosa.feature.rms(y=y, hop_length=256).flatten()
        rms_medio_saltos = float(rms_saltos.mean()) if len(rms_saltos) > 0 else 1.0

        centroide = librosa.feature.spectral_centroid(y=y, sr=sr).flatten()
        centroide_medio = float(centroide.mean()) if len(centroide) > 0 else 1.0
        
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Retornamos los diccionarios empaquetados para mantener limpio el flujo
        dsp_rms_data = {
            "bpm": (rms_bpm, rms_medio_bpm, 1024),
            "speed": (rms_speed, rms_medio_speed, 512),
            "saltos": (rms_saltos, rms_medio_saltos, 256)
        }
        
        return bpm_detectado, duracion_segundos, dsp_rms_data, centroide, centroide_medio, mel_db

    except Exception as e:
        print(f"Error en análisis DSP: {e}")
        # Caída de seguridad pasiva
        vacio = np.array([])
        fallback = {"bpm": (vacio, 1.0, 1024), "speed": (vacio, 1.0, 512), "saltos": (vacio, 1.0, 256)}
        return 120.0, 180.0, fallback, np.array([]), 1.0, None

def detectar_primer_drop_rms(rms, sr, hop_length=512, umbral_porcentaje=0.05):
    """
    Detecta el segundo exacto del impacto rítmico inicial.
    CORREGIDO: Extracción de índices planos con np.flatnonzero para evitar errores de tipo tupla.
    """
    if len(rms) == 0:
        return 0.000

    # 1. Establecer el suelo de ruido analizando los frames iniciales más calmados
    suelo_ruido = np.percentile(rms, 15)  
    rms_maximo = np.max(rms)
    
    # 2. El umbral combina el porcentaje del pico máximo y el suelo base del archivo
    umbral_energia = suelo_ruido + (rms_maximo - suelo_ruido) * umbral_porcentaje
    
    # 3. Aplicar filtro de gradiente (Derivada) para buscar subidas abruptas de energía
    rms_diff = np.diff(rms, prepend=0)
    umbral_cambio = np.std(rms_diff) * 0.5  # Sensibilidad al cambio de transitorios
    
    # [PARCHE CRÍTICO]: np.flatnonzero extrae directamente los índices enteros planos sin tuplas problemáticas
    indices_candidatos = np.flatnonzero((rms > umbral_energia) & (rms_diff > umbral_cambio))
    
    if len(indices_candidatos) > 0:
        primer_frame = int(indices_candidatos[0])  # Conversión segura a entero de Python
        tiempo_segundos = librosa.frames_to_time(primer_frame, sr=sr, hop_length=hop_length)
        return round(float(tiempo_segundos), 3)
        
    # Caída de seguridad pasiva por si fallan los diferenciales de gradiente abruptos
    indices_pasivos = np.flatnonzero(rms > umbral_energia)
    if len(indices_pasivos) > 0:
        primer_frame = int(indices_pasivos[0])  # Conversión segura a entero de Python
        tiempo_segundos = librosa.frames_to_time(primer_frame, sr=sr, hop_length=hop_length)
        return round(float(tiempo_segundos), 3)

    return 0.000

def procesar_ritmo_dinamico(segundo_actual, beat_actual, paso_idx, lineas_objetivo, rms_bpm, rms_medio_bpm, hop_bpm, ultimo_bpm, config_bpm):
    """
    Fase 0A: Evalúa la tendencia macro del RMS y altera el tempo de forma progresiva (Efecto Marea).
    Aplica un suavizado de ventana sobre el RMS e interpolación hacia el BPM objetivo.
    Garantiza que el BPM calculado no sea cero, previniendo errores de división posteriores.
    """

    # --- CONSTANTE DE SEGURIDAD ---
    # Definimos un límite inferior para cualquier BPM devuelto (Ej: 20 BPM)
    MIN_BPM = max(5, np.abs(config_bpm["bajo"]) * 0.5) # Aseguramos que sea positivo y razonable

    if len(rms_bpm) == 0 or paso_idx % max(1, lineas_objetivo // 4) != 0:
        # Si no hay datos de entrada o es un paso que no debe calcularse, retornamos el último valor.
        return ultimo_bpm, None

    f_idx = min(int((segundo_actual * 22050) / hop_bpm), len(rms_bpm) - 1)
    
    # --- FILTRO DE MAREA 1: Promedio móvil local (Ventana de 5 frames para suavizar picos transitorios)
    f_inicio = max(0, f_idx - 2)
    f_fin = min(len(rms_bpm), f_idx + 3)
    rms_suavizado = np.mean(rms_bpm[f_inicio:f_fin])
    
    # Manejo de división por cero en el ratio
    ratio = rms_suavizado / rms_medio_bpm if rms_medio_bpm > 1e-6 else 1.0
    
    r_min = config_bpm["r_min"]
    r_med = config_bpm["r_med"]
    umbral_c = config_bpm["umbral_c"]
    umbral_d = config_bpm["umbral_d"]
    
    # Determinar el BPM ideal al que la música "quiere" llegar
    if ratio <= r_min:
        bpm_objetivo = config_bpm["bajo"]
    elif ratio <= r_med:
        bpm_objetivo = config_bpm["media_baja"] if ratio <= umbral_c else config_bpm["medio"]
    else:
        bpm_objetivo = config_bpm["media_alta"] if ratio <= umbral_d else config_bpm["alto"]

    # --- FILTRO DE MAREA 2: Interpolación Lineal (Lerp)
    # En lugar de saltar directo al objetivo, nos acercamos un 12% en cada evaluación.
    # Esto genera una aceleración/deceleración orgánica ("marea").
    # --- LECTURA DEL AMORTIGUADOR PERSONALIZADO DESDE LA GUI ---
    factor_amortiguacion = config_bpm["amortiguador_custom"]
    bpm_calc = ultimo_bpm + (bpm_objetivo - ultimo_bpm) * factor_amortiguacion

    # Aplicar la restricción de seguridad: el BPM calculado debe ser al menos MIN_BPM
    if bpm_calc < MIN_BPM:
        bpm_calc = MIN_BPM

    # Filtro de histéresis: Solo registramos el cambio en el Simfile si la marea se desplazó más de 1.5 BPM
    if abs(bpm_calc - ultimo_bpm) > 1.5:
        return round(bpm_calc, 3), f"{beat_actual:.3f}={bpm_calc:.3f}"
        
    return ultimo_bpm, None

def procesar_scroll_dinamico(segundo_actual, beat_actual, paso_idx, lineas_objetivo, rms_speed, rms_medio_speed, hop_speed, ultima_vel, config_speed):
    """
    Fase 0B: Modifica la velocidad visual delegando la transición nativa a StepMania (.SSC).
    Usa el parámetro 'trans' configurado por el usuario en la GUI para controlar la duración 
    exacta de la rampa musical (en Beats) sin cortes de mood.
    """
    # Regulamos la lectura: evaluamos cada inicio de cuarto de compás (nota negra)
    # para dar tiempo a que las transiciones nativas respiren y se muestren completas.
    if len(rms_speed) == 0 or paso_idx % max(1, lineas_objetivo // 4) != 0:
        return ultima_vel, None

    f_idx = min(int((segundo_actual * 22050) / hop_speed), len(rms_speed) - 1)
    
    v_min = config_speed["v_min"]
    v_max = config_speed["v_max"]
    r_min = config_speed["r_min"]
    r_max = config_speed["r_max"]
    r_diff = config_speed["r_diff"]
    duracion_transicion_beats = config_speed["speed_trans_custom"]
    umbral_disparo = config_speed["umbral_disparo_custom"]

    if ultima_vel < 0:
        ultima_vel = v_min

    # Filtro de suavizado espectral base
    f_inicio = max(0, f_idx - 3)
    f_fin = min(len(rms_speed), f_idx + 4)
    rms_suavizado = np.mean(rms_speed[f_inicio:f_fin])
    
    ratio = rms_suavizado / rms_medio_speed if rms_medio_speed > 0 else 1.0
    
    # Mapeo lineal directo de los picos de audio a los límites de velocidad del usuario
    factor = max(0.0, min((ratio - r_min) / r_diff, 1.0))
    vel_objetivo = v_min + factor * (v_max - v_min)
    
    # Redondeo para estabilizar lecturas y evitar fluctuaciones decimales insignificantes
    vel_objetivo = round(vel_objetivo, 2)

    # --- DISPARADOR NATIVO POR UMBRAL DE CAMBIO ---
    # Solo emitimos un comando #SPEEDS si el audio cambió de forma perceptible (> umbral_disparox).
    # Esto evita que StepMania interrumpa una transición que ya está en curso, logrando que las
    # aceleraciones y frenados nativos se ejecuten de manera limpia y espectacular en pantalla.
    if abs(vel_objetivo - ultima_vel) > umbral_disparo:
        # Formato Nativo de StepMania .SSC: Beat=Velocidad=DuraciónEnBeats=Modo(0=Lineal)
        comando_ssc = f"{beat_actual:.3f}={vel_objetivo:.3f}={duracion_transicion_beats:.3f}=0"
        return vel_objetivo, comando_ssc
        
    return ultima_vel, None

def inyectar_saltos_espectrales_paso(paso_elegido, ratio_energia, bpm, lineas_por_compas, dificultad_tag, custom_params):
    """
    Fase 2 (Optimizada): Convierte notas simples ('1') en saltos dobles ante transitorios rápidos.
    Retorna el paso modificado y el número de líneas consecutivas que se deben limpiar (cooldown).
    """
    if not custom_params.get("activar_secciones_saltos", False):
        return paso_elegido, 0

    r_min = custom_params.get("saltos_rms_min", 0.5)
    r_max = custom_params.get("saltos_rms_max", 1.5)
    r_diff = (r_max - r_min) if (r_max - r_min) != 0 else 1.0
    
    # Umbral adaptativo dinámico basado en las preferencias de la GUI
    umbral_dinamico_corte = r_min + (r_diff * 0.85)

    if ratio_energia > umbral_dinamico_corte:
        # Solo actuar si es una nota simple física ('1') libre de Holds o Roll heads
        if paso_elegido.count('1') == 1 and paso_elegido.count('2') == 0 and paso_elegido.count('4') == 0:
            fila_lista = list(paso_elegido)
            columnas_vacias = [col for col, char in enumerate(fila_lista) if char == '0']
            
            if columnas_vacias:
                # 1. Inyección del salto doble aleatorio
                fila_lista[random.choice(columnas_vacias)] = '1'
                paso_modificado = "".join(fila_lista)
                
                # 2. Cálculo dinámico de la cascada de borrado (anti-saturación)
                proporcion_borrado = 0.250 if bpm > 180 else (0.125 if bpm < 100 else 0.187)
                pasos_a_borrar = max(1, int(round(lineas_por_compas * proporcion_borrado)))
                
                return paso_modificado, pasos_a_borrar

    return paso_elegido, 0

def inyectar_trampas_sincronas(paso_elegido, ratio_energia, cfg, custom_params, conteo_compas_fx):
    """
    Fase 3 (RECALIBRADA): Reemplaza notas físicas de forma aleatoria por Minas o FX avanzados.
    Utiliza un diccionario de control de presupuesto por compás para evitar la saturación visual.
    """
    if '1' not in paso_elegido:
        return paso_elegido

    tipos_modificadores = [
        {"char": "M", "key": "max_minas_compas", "prob": cfg.get("probabilidad_minas", 0.0)},
        {"char": "F", "key": "max_fakes_compas", "prob": cfg.get("probabilidad_fakes", 0.0)},
        {"char": "L", "key": "max_lifts_compas", "prob": cfg.get("probabilidad_lifts", 0.0)},
        {"char": "P", "key": "max_potions_compas", "prob": cfg.get("probabilidad_potions", 0.0)},
        {"char": "D", "key": "max_shields_compas", "prob": cfg.get("probabilidad_shields", 0.0)},
        {"char": "S", "key": "max_rayos_compas", "prob": cfg.get("probabilidad_rayos", 0.0)},
        {"char": "H", "key": "max_hiddens_compas", "prob": cfg.get("probabilidad_hiddens", 0.0)}
    ]

    r_min = custom_params.get("fx_rms_min", 0.5)
    r_max = custom_params.get("fx_rms_max", 1.5)
    r_diff = (r_max - r_min) if (r_max - r_min) != 0 else 1.0

    # Mezclar el orden de evaluación para que un modificador no tenga siempre prioridad sobre otro
    random.shuffle(tipos_modificadores)

    for n_type in tipos_modificadores:
        char = n_type["char"]
        max_permitido = cfg.get(n_type["key"], 0)
        
        # Si el límite configurado es 0 o ya alcanzamos el presupuesto máximo en este compás, saltar
        if max_permitido <= 0 or conteo_compas_fx.get(char, 0) >= max_permitido:
            continue

        # Escalado de intensidad basado en el audio
        factor_intensidad = max(0.0, min((ratio_energia - r_min) / r_diff, 1.0))
        prob_final = min(1.0, n_type["prob"] * (1.0 + factor_intensidad))
        
        # Reducción drástica de probabilidad si se intentan encadenar modificadores idénticos seguidos
        # (Evita ráfagas desagradables de minas o fakes seguidas)
        if conteo_compas_fx.get(char, 0) > 0:
            prob_final *= 0.35 

        if random.random() < prob_final:
            fila_lista = list(paso_elegido)
            indices_flecha = [col for col, char_f in enumerate(fila_lista) if char_f == '1']
            
            if indices_flecha:
                col_elegida = random.choice(indices_flecha)
                fila_lista[col_elegida] = char
                paso_elegido = "".join(fila_lista)
                
                # Registramos el uso en el inventario del compás actual
                conteo_compas_fx[char] = conteo_compas_fx.get(char, 0) + 1
                break # Solo permitimos transformar una flecha por línea para mantener la cordura anatómica
                
    return paso_elegido


def ejecutar_bucle_sincrono(config_dificultad, compases_totales, bpm, val_offset, duracion, total_frames,
                            dinamico_activo, speeds_activo, custom_params, aplicar_post, max_level_chosen,
                            rms_bpm, rms_medio_bpm, hop_bpm, rms_speed, rms_medio_speed, hop_speed,
                            rms_saltos, rms_medio_saltos, hop_saltos, lista_rms, rms_medio, lista_centroide,
                            centroide_medio, mel_db, modelo, dispositivo, tokenizer, PostProcesadorStepMania):
    """
    Bucle principal reestructurado para la generación híbrida de flechas en tiempo real.
    Unifica la inferencia de la IA con el análisis DSP paso a paso para garantizar sincronía.
    """
    
    # Inicialización de buffers de control rítmico globales
    lista_cambios_bpm = [f"0.000={bpm:.3f}"]
    lista_cambios_speeds = []
    mapa_pasos_por_dificultad = {}
    log_metricas_diff = ""

    # Diccionarios de empaquetado para configuraciones rítmicas (Evitan re-calcular en el ciclo)
    dicc_config_bpm = {}
    if dinamico_activo:
        min_bpm_dinamico = custom_params.get("min_bpm_dinamico", 0)
        max_bpm_dinamico = custom_params.get("max_bpm_dinamico", 0)
        if min_bpm_dinamico == 0 and max_bpm_dinamico == 0:
            min_bpm_dinamico, max_bpm_dinamico = bpm - 30, bpm + 30
        valores_ordenados = sorted([max(0.001, min_bpm_dinamico), bpm, max(0.001, max_bpm_dinamico)])
        BPM_ESTRATO_BAJO, BPM_ESTRATO_MEDIO, BPM_ESTRATO_ALTO = valores_ordenados[0], valores_ordenados[1], valores_ordenados[2]
        media_baja = BPM_ESTRATO_BAJO + (BPM_ESTRATO_MEDIO - BPM_ESTRATO_BAJO) * 0.75
        media_alta = BPM_ESTRATO_MEDIO + (BPM_ESTRATO_ALTO - BPM_ESTRATO_MEDIO) * 0.75
        r_min_bpm = custom_params.get("bpm_dinamico_rms_min", 0.5)
        r_max_bpm = custom_params.get("bpm_dinamico_rms_max", 1.5)
        r_medio_bpm = r_min_bpm + (r_max_bpm - r_min_bpm) / 2
        umbral_calma, umbral_drop = r_min_bpm + (r_medio_bpm - r_min_bpm) / 2, r_medio_bpm + (r_max_bpm - r_medio_bpm) / 2
        
        dicc_config_bpm = {
            "bajo": BPM_ESTRATO_BAJO, "medio": BPM_ESTRATO_MEDIO, "alto": BPM_ESTRATO_ALTO,
            "media_baja": media_baja, "media_alta": media_alta, "r_min": r_min_bpm, 
            "r_med": r_medio_bpm, "umbral_c": umbral_calma, "umbral_d": umbral_drop,
            "amortiguador_custom": custom_params.get("bpm_amortiguador_custom", 0.12) 
        }

    dicc_config_speed = {}
    if speeds_activo:
        VELOCIDAD_MINIMA = custom_params.get("speed_min_custom", 0.70)
        VELOCIDAD_MAXIMA = custom_params.get("speed_max_custom", 1.40)
        duracion_transicion = custom_params.get("speed_trans_custom", 2.0)
        r_min_speed = custom_params.get("speed_rms_min", 0.5)
        r_max_speed = custom_params.get("speed_rms_max", 1.5)
        r_diff_speed = (r_max_speed - r_min_speed) if (r_max_speed - r_min_speed) != 0 else 1.0
        
        dicc_config_speed = {
            "v_min": VELOCIDAD_MINIMA, "v_max": VELOCIDAD_MAXIMA, "trans": 0.0,
            "r_min": r_min_speed, "r_max": r_max_speed, "r_diff": r_diff_speed,
            "speed_trans_custom": custom_params.get("speed_trans_custom", 0.05) ,
            "umbral_disparo_custom": custom_params.get("speed_umbral_disparo", 0.50)
        }
    # Bucle por cada dificultad configurada
    for diff, cfg in config_dificultad.items():
        pasos_finales_ia = []
        lineas_objetivo = cfg["lineas_por_compas"]

        porcentaje = custom_params.get("porcentaje_tamano", 1.0)

        total_pasos_dificultad = int((compases_totales * lineas_objetivo) * porcentaje)
        
        segundo_actual = val_offset 
        beat_actual = 0.0
        ultimo_bpm_aplicado = bpm
        ultima_velocidad_aplicada = -1.0

        cooldown_saltos = 0 

        # Diccionario de control para el presupuesto de trampas de este compás
        conteo_compas_fx = {"M": 0, "F": 0, "L": 0, "P": 0, "D": 0, "S": 0, "H": 0}

        for paso_idx in range(total_pasos_dificultad):
            beats_por_paso = 4.0 / lineas_objetivo
            
            # --- REINICIO AUTOMÁTICO DE PRESUPUESTO POR COMPÁS ---
            if paso_idx % lineas_objetivo == 0:
                conteo_compas_fx = {"M": 0, "F": 0, "L": 0, "P": 0, "D": 0, "S": 0, "H": 0}

            # -----------------------------------------------------------------
            # FASE 0: VALIDACIÓN DE TEMPO Y VELOCIDADES (CHECKBOXES DE RITMO)
            # -----------------------------------------------------------------
            if dinamico_activo:
                #from dsp_synchronous_engine import procesar_ritmo_dinamico
                ultimo_bpm_aplicado, cambio_bpm_str = procesar_ritmo_dinamico(
                    segundo_actual, beat_actual, paso_idx, lineas_objetivo, 
                    rms_bpm, rms_medio_bpm, hop_bpm, ultimo_bpm_aplicado, dicc_config_bpm
                )
                if cambio_bpm_str:
                    lista_cambios_bpm.append(cambio_bpm_str)

            if speeds_activo:
                #from dsp_synchronous_engine import procesar_scroll_dinamico
                ultima_velocidad_aplicada, cambio_speed_str = procesar_scroll_dinamico(
                    segundo_actual, beat_actual, paso_idx, lineas_objetivo, 
                    rms_speed, rms_medio_speed, hop_speed, ultima_velocidad_aplicada, dicc_config_speed
                )
                if cambio_speed_str:
                    lista_cambios_speeds.append(cambio_speed_str)

            segundos_por_paso = (60.0 / ultimo_bpm_aplicado) * beats_por_paso

            # -----------------------------------------------------------------
            # FASE 1: INFERENCIA DE LA IA (Transformador Causal)
            # -----------------------------------------------------------------
            frame_idx = min(int((segundo_actual / duracion) * total_frames), total_frames - 1)
            
            # Control adaptativo de densidad nativo de la IA
            # Recuperamos el estado del checkbox de saltos desde los parámetros del usuario
            saltos_activos = custom_params.get("activar_secciones_saltos", False)

            # (Solo si el checkbox está encendido)
            if saltos_activos and cooldown_saltos > 0:
                paso_elegido = "0000"
                cooldown_saltos -= 1  # Decrementar el enfriamiento de forma segura paso a paso
            else:
                # Si desactivan el checkbox, purgamos cualquier cooldown colgado
                cooldown_saltos = 0 
                
                # Control de densidad nativo de la IA para notas consecutivas
                if paso_idx > 0 and pasos_finales_ia and pasos_finales_ia[-1] != "0000":
                    limite_densidad = {"Easy": 0.40, "Medium": 0.30, "Hard": 0.15, "Challenge": 0.05}
                    if random.random() < limite_densidad[diff]:
                        paso_elegido = "0000"
                    else:
                        paso_elegido = None
                else:
                    paso_elegido = None

            if paso_elegido is None:
                if paso_idx < 2:
                    paso_elegido = "1000"
                else:
                    bloque_contexto = []
                    for offset in range(-127, 1):
                        f_rms_ia = max(0, min(frame_idx + offset, len(lista_rms) - 1))
                        f_centroide = max(0, min(frame_idx + offset, len(lista_centroide) - 1))
                        r_energia = lista_rms[f_rms_ia] / rms_medio if len(lista_rms) > 0 else 1.0
                        r_brillo = lista_centroide[f_centroide] / centroide_medio if len(lista_centroide) > 0 else 1.0
                        
                        if mel_db is not None and mel_db.shape[1] > 0:
                            f_mel = max(0, min(frame_idx + offset, mel_db.shape[1] - 1))
                            vector_mel = mel_db[:, f_mel].tolist()
                        else:
                            vector_mel = [0.0] * 64
                        bloque_contexto.append([r_energia, r_brillo] + vector_mel)

                    tensor_in = torch.tensor(bloque_contexto, dtype=torch.float32).unsqueeze(0).to(dispositivo)
                    with torch.no_grad():
                        logits = modelo(tensor_in)[:, -1, :]
                        probabilidades = torch.softmax(logits / cfg["temperatura"], dim=-1)
                        prediccion = torch.multinomial(probabilidades, num_samples=1).item()
                    paso_elegido = "".join(tokenizer.decode_id(prediccion))

            # -----------------------------------------------------------------
            # FASE 2 & 3: COMPLEMENTOS DSP SÍNCRONOS (CHECKBOXES DE EFECTOS)
            # -----------------------------------------------------------------
            f_idx_saltos = max(0, min(int((segundo_actual * 22050) / hop_saltos), len(rms_saltos) - 1))
            ratio_energia_paso = rms_saltos[f_idx_saltos] / rms_medio_saltos if rms_medio_saltos > 0 else 1.0

            if saltos_activos:
                #from dsp_synchronous_engine import inyectar_saltos_sincronos
                # Ejecución de la nueva función reconstructiva de saltos
                paso_elegido, pasos_limpieza = inyectar_saltos_espectrales_paso(
                    paso_elegido=paso_elegido, 
                    ratio_energia=ratio_energia_paso, 
                    bpm=ultimo_bpm_aplicado, 
                    lineas_por_compas=lineas_objetivo, 
                    dificultad_tag=diff, # <--- Enviamos la dificultad actual ("Hard", "Challenge", etc.)
                    custom_params=custom_params
                )
                
                # Si se generó un salto exitoso, cargamos el cooldown para las iteraciones siguientes
                if pasos_limpieza > 0:
                    cooldown_saltos = pasos_limpieza

            if custom_params.get("efectos_por_rms", False):
                #from dsp_synchronous_engine import inyectar_trampas_sincronas
                if custom_params.get("efectos_por_rms", False):
                    paso_elegido = inyectar_trampas_sincronas(
                        paso_elegido=paso_elegido, 
                        ratio_energia=ratio_energia_paso, 
                        cfg=cfg, 
                        custom_params=custom_params,
                        conteo_compas_fx=conteo_compas_fx 
                    )

            # Acumulación en el buffer temporal y avance de relojes síncronos
            pasos_finales_ia.append(paso_elegido)
            segundo_actual += segundos_por_paso
            beat_actual += beats_por_paso

        # -----------------------------------------------------------------
        # FASE 4: EXTENSIÓN ESTÉTICA Y CORRECCIÓN ANATÓMICA
        # -----------------------------------------------------------------
        extension_final = custom_params.get("extension_final", 0.0) if custom_params else 0.0
        if extension_final > 0:
            compases_extras = math.ceil(extension_final / ((60.0 / ultimo_bpm_aplicado) * 4.0))
            for l_idx in range(compases_extras * lineas_objetivo):
                if l_idx == (compases_extras * lineas_objetivo) - 1:
                    pasos_finales_ia.append("V000")
                else:
                    pasos_finales_ia.append("D00D")

        if aplicar_post:
            pasos_finales_ia = PostProcesadorStepMania.corregir_sintaxis_holds(
                pasos_finales_ia, 
                pasos_maximos_hold=cfg["max_hold"], 
                lineas_por_compas=lineas_objetivo,
                max_holds_simultaneos=custom_params.get("holds_simultaneos", 2),
            )

        # Formateo de los bloques de medidas rítmicas del Simfile
        bloque_pasos_texto = ""
        compas_print = []
        for idx, paso in enumerate(pasos_finales_ia):
            compas_print.append(paso)
            if len(compas_print) == lineas_objetivo or idx == len(pasos_finales_ia) - 1:
                bloque_pasos_texto += "\n".join(compas_print)
                bloque_pasos_texto += "\n;\n" if idx == len(pasos_finales_ia) - 1 else "\n,\n"
                compas_print = []

        mapa_pasos_por_dificultad[diff] = bloque_pasos_texto

        # Recálculo dinámico del METER real adaptado al tamaño de la muestra
        if custom_params.get("recalcular_dificultad", False):
            # Extraemos el porcentaje actual para castigar el NPS proporcionalmente
            pct_actual = custom_params.get("porcentaje_tamano", 1.0)
            
            meter_real, texto_reporte = PostProcesadorStepMania.recalcular_meter_real(
                secuencia_pasos=pasos_finales_ia, duracion_segundos=duracion, dificultad_tag=diff,
                max_level_chosen=max_level_chosen, max_notas_compas=cfg.get("max_notas_compas", 12),
                extension_final=extension_final, custom_params=custom_params, lineas_por_compas=lineas_objetivo, bpm=bpm,
                factor_escala_muestreo=pct_actual 
            )
            config_dificultad[diff]["meter"] = meter_real
            log_metricas_diff += texto_reporte

    # Purgar duplicados de las cadenas de sincronización globales
    bpms_string_line = ",\n".join(dict.fromkeys(lista_cambios_bpm))
    speeds_string_line = ",\n".join(dict.fromkeys(lista_cambios_speeds)) if lista_cambios_speeds else ""

    return mapa_pasos_por_dificultad, bpms_string_line, speeds_string_line, log_metricas_diff


# =====================================================================
# CORE DE GENERACIÓN HÍBRIDA DUAL (.SM y .SSC)
# =====================================================================
def generar_simfiles_hibridos(audio_path, checkpoint_path, song_title, max_level_chosen, 
                              carpeta_salida, artist_name="", banner_path="", video_path="", duracion_limite=0.0, custom_params=None):

    # Recuperar y fijar la semilla de forma estricta antes de que la IA o el DSP hagan algo
    seed_actual = custom_params.get("seed_value", 42)
    fijar_semilla_determinista(seed_actual)

    config_dificultad = {
        "Easy": {
            "temperatura": custom_params["temperatura"] if custom_params else 0.7, 
            "max_hold": custom_params["max_hold"] if custom_params else 4, 
            "meter": max(1, int(max_level_chosen * 0.25)),
            "lineas_por_compas": max(2, int(custom_params["lineas_por_compas"] * 0.5)) if custom_params else 8, 
            "min_notas_compas": max(2, int(custom_params["min_notas_compas"] * 0.5)) if custom_params else 4, 
            "max_notas_compas": max(2, int(custom_params["max_notas_compas"] * 0.5)) if custom_params else 9,
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
            "lineas_por_compas": max(2, int(custom_params["lineas_por_compas"] * 0.66)) if custom_params else 8, 
            "min_notas_compas": max(2, int(custom_params["min_notas_compas"] * 0.66)) if custom_params else 4, 
            "max_notas_compas": max(2, int(custom_params["max_notas_compas"] * 0.66)) if custom_params else 9,
            "max_minas_compas": min(1, custom_params["max_minas_compas"]) if custom_params else 1, 
            "probabilidad_minas": (custom_params["probabilidad_minas"] * 0.5) if custom_params else 0.15,
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
            "lineas_por_compas": max(2, int(custom_params["lineas_por_compas"] * 0.9)) if custom_params else 12, 
            "min_notas_compas": max(2, int(custom_params["min_notas_compas"] * 0.75)) if custom_params else 4, 
            "max_notas_compas": max(2, int(custom_params["max_notas_compas"] * 0.75)) if custom_params else 9,
            "max_minas_compas": min(2, custom_params["max_minas_compas"]) if custom_params else 2, 
            "probabilidad_minas": (custom_params["probabilidad_minas"] * 0.75) if custom_params else 0.25,
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
            "lineas_por_compas": custom_params["lineas_por_compas"] if custom_params else 12, 
            "min_notas_compas": custom_params["min_notas_compas"] if custom_params else 8, 
            "max_notas_compas": custom_params["max_notas_compas"] if custom_params else 12, 
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

    # Invocamos el nuevo motor DSP asimétrico
    bpm, duracion, dsp_rms, lista_centroide, centroide_medio, mel_db = analizar_audio_hibrido(audio_path)

    # Desempaquetamos los buffers de energía específicos
    rms_bpm, rms_medio_bpm, hop_bpm = dsp_rms["bpm"]
    rms_speed, rms_medio_speed, hop_speed = dsp_rms["speed"]
    rms_saltos, rms_medio_saltos, hop_saltos = dsp_rms["saltos"]

    # =====================================================================
    # 🔥 SOLUCIÓN CRÍTICA: Vinculación de variables para evitar NameError
    # =====================================================================
    lista_rms = rms_saltos        # Asignamos el RMS micro para la densidad de notas de la IA
    rms_medio = rms_medio_saltos  # Su respectiva media global

    speeds_activo = custom_params.get("aplicar_speeds_dinamicos", False) if custom_params else False
    # Verificamos si existe una ampliación en la duración por aplicar scroll speeds
    percent_speed_offset_time = custom_params["speed_offset_time"] if custom_params and speeds_activo else 0
    
    # Si utilizamos la duración dada por el usuario > 0
    if duracion_limite > 0.0:
        #Verificamos si se están utilizando scrolls speeds
        if speeds_activo:
            speed_offset_time = 0.1 * percent_speed_offset_time * duracion_limite
            duracion = duracion_limite + speed_offset_time
        else:
            duracion = duracion_limite
    else: # En caso contrario aumentamos la duración del audio más el aumento por pérdida
        if speeds_activo:
            speed_offset_time = 0.1 * percent_speed_offset_time * duracion
            duracion = duracion + speed_offset_time
        else: #Redundante pero se entiende que sino existen scrolls se toma la duración auto
            duracion = duracion
    
    double_bpm_factor = 2.0 if (custom_params and custom_params.get("double_bpm") == True) else 1.0

    #Aplicamos un bpm manual si es configurado desde la interfaz
    if custom_params and custom_params.get("bpm_automatico") == False:
        bpm = custom_params.get("bpm_manual")

    if double_bpm_factor > 1.0:
        bpm *= double_bpm_factor

    # Extraer el offset numérico de la interfaz gráfica
    # --- CÁLCULO DE OFFSET AUTOMÁTICO VS MANUAL ---
    if custom_params.get("offset_automatico", False):
        # Detectamos el offset real analizando el RMS. 
        # Librosa por defecto usa un hop_length de 256 en librosa.feature.rms
        val_offset = detectar_primer_drop_rms(lista_rms, sr=22050, hop_length=hop_saltos)
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

        # --- LÓGICA DE COEXISTENCIA PARA BPMS Y SPEEDS DINÁMICOS ---
    dinamico_activo = custom_params.get("aplicar_bpm_dinamico", False) if custom_params else False

    post = PostProcesadorStepMania()

    #----------------------------------------------------
    # Procesamiento de archivos multimedia
    #----------------------------------------------------

    nombre_audio = os.path.basename(audio_path)
    #Verificacion de seguridad para el banner y video
    nombre_banner = os.path.basename(banner_path) if banner_path else ""
    nombre_video = os.path.basename(video_path) if video_path else ""

    # Definir la jerarquía de carpetas estándar: Raíz -> Nombre del Pack -> Nombre de la Canción
    nombre_grupo = custom_params.get("pack_name", "AI_Generated_Charts")
    folder_song_sanitizada = "".join([c for c in song_title if c.isalnum() or c in [' ', '_', '-']]).strip()
    folder_group_sanitizada = "".join([c for c in nombre_grupo if c.isalnum() or c in [' ', '_', '-']]).strip()
        
    carpeta_raiz_packs = os.path.join(carpeta_salida, "AI_Generated_Packs")
    carpeta_pack_final = os.path.join(carpeta_raiz_packs, folder_group_sanitizada, folder_song_sanitizada)

    # Crear de forma segura todo el árbol de directorios requerido
    os.makedirs(carpeta_pack_final, exist_ok=True)

    # Procesamiento y renombrado seguro del archivo de Audio (.mp3/.wav)
    if custom_params.get("renombrar_archivos", False):
        _, extension = os.path.splitext(nombre_audio)
        new_audio_name = f"{folder_song_sanitizada}{extension}"
    else:
        new_audio_name = nombre_audio
        
    ruta_final_audio = os.path.join(carpeta_pack_final, new_audio_name)
    if os.path.abspath(audio_path) != os.path.abspath(ruta_final_audio):
        shutil.copy(audio_path, ruta_final_audio)

    # Copiar Banner y Video únicamente verificando copias
    if banner_path and os.path.exists(banner_path):
        if custom_params.get("renombrar_archivos", False):
            _, extension = os.path.splitext(nombre_banner)
            new_banner_name = f"{folder_song_sanitizada}{extension}"
        else:
            new_banner_name = nombre_banner
        ruta_final_banner = os.path.join(carpeta_pack_final, new_banner_name)
        if os.path.abspath(banner_path) != os.path.abspath(ruta_final_banner):
            shutil.copy(banner_path, ruta_final_banner)
    else:
        new_banner_name = ""

    if video_path and os.path.exists(video_path):
        if custom_params.get("renombrar_archivos", False):
            _, extension = os.path.splitext(nombre_video)
            new_video_name = f"{folder_song_sanitizada}{extension}"
        else:
            new_video_name = nombre_video
        ruta_final_video = os.path.join(carpeta_pack_final, new_video_name)
        if os.path.abspath(video_path) != os.path.abspath(ruta_final_video):
            shutil.copy(video_path, ruta_final_video)
    else:
        new_video_name = ""

    bg_changes_line = f"0.000={new_video_name}=1.000=1=0=0=crossfade=," if new_video_name else ""

    if seed_actual < 100000:
        emoji = "👌" 
    elif seed_actual < 200000:
        emoji = "🐞" 
    elif seed_actual < 300000:
        emoji = "🍋" 
    elif seed_actual < 400000:
        emoji = "🗿"
    elif seed_actual < 500000:
        emoji = "🌝" 
    elif seed_actual < 600000:
        emoji = "🌟" 
    elif seed_actual < 700000:
        emoji = "👺" 
    elif seed_actual < 800000:
        emoji = "🐟"
    elif seed_actual < 900000:
        emoji = "🍥"
    else:
        emoji = "🎲​"

    #Indicador adicional de dificultad
    subtitulo_dev = f"{emoji}​ DIFICULTAD ÓPTIMA"

    if custom_params["max_notas_compas"] > 16 and custom_params["max_notas_compas"] <= 32:
        subtitulo_dev = "⚠️ HARD DIFFICULT, EXPONENTIAL DENSITY" 
    elif custom_params["max_notas_compas"] > 32:
        subtitulo_dev = "💀​ HARDCORE DIFFICULT, MAXIMUM DENSITY"

    # ---------------------------------------------------------------------
    # SISTEMA DE MUESTREO MULTI-ARCHIVO ADAPTATIVO CON LINSPACE
    # ---------------------------------------------------------------------
    activar_muestreo = custom_params.get("activar_muestreo_multicapa", False)
    
    if activar_muestreo:
        porcentaje_min = custom_params.get("muestreo_pct_min", 0.95)
        num_muestras = custom_params.get("muestreo_num_muestras", 7)
        # Genera puntos de muestreo distribuidos de manera equidistante hasta el 1.00 (100%)
        porcentajes_muestreo = np.linspace(porcentaje_min, 1.00, num=num_muestras).tolist()
    else:
        # Caída por defecto si el checkbox está desactivado: solo genera el archivo estándar al 100%
        porcentajes_muestreo = [1.00]

    log_metricas_acumulado = ""
    
    for idx_m, pct in enumerate(porcentajes_muestreo):
        custom_params["porcentaje_tamano"] = pct
        # Generar las matrices de pasos reducidas/completas de forma síncrona
        mapa_pasos_por_dificultad, bpms_string_line, speeds_string_line, log_metricas_diff = ejecutar_bucle_sincrono(config_dificultad=config_dificultad, compases_totales=compases_totales,
            bpm=bpm, val_offset=val_offset, duracion=duracion, total_frames=total_frames, dinamico_activo=dinamico_activo, speeds_activo=speeds_activo, custom_params=custom_params,
            aplicar_post=aplicar_post, max_level_chosen=max_level_chosen, rms_bpm=rms_bpm, rms_medio_bpm=rms_medio_bpm, hop_bpm=hop_bpm, rms_speed=rms_speed, rms_medio_speed=rms_medio_speed,
            hop_speed=hop_speed, rms_saltos=rms_saltos, rms_medio_saltos=rms_medio_saltos, hop_saltos=hop_saltos, lista_rms=lista_rms, rms_medio=rms_medio, lista_centroide=lista_centroide,
            centroide_medio=centroide_medio, mel_db=mel_db, modelo=modelo, dispositivo=dispositivo, tokenizer=tokenizer, PostProcesadorStepMania=post)

    # =====================================================================
    # 📁 PLANTILLA DE EMPAQUETADO AUTOMÁTICO PARA STEPMANIA / OUTFOX
    # =====================================================================
        # --- ASIGNACIÓN DE SUFIJOS DE SALIDA SEGÚN CONFIGURACIÓN ---
        if activar_muestreo:
            sufijo_tamano = f"size_{int(round(pct*100))}"
            titulo_simfile = f"{song_title} ({int(round(pct*100))}% Size)"
            subtitulo_muestra = f"🎲 CAPA DE MUESTREO: {int(round(pct*100))}%"
            subtitulo = f"{subtitulo_muestra} {subtitulo_dev}"
        else:
            sufijo_tamano = "original"
            titulo_simfile = song_title
            subtitulo = subtitulo_dev

        name_output = f"{folder_song_sanitizada}_{seed_actual}_{sufijo_tamano}"
        output_sm = os.path.join(carpeta_pack_final, f"{name_output}.sm")
        output_ssc = os.path.join(carpeta_pack_final, f"{name_output}.ssc")

        # --- ESCRITURA DEL ARCHIVO .SM INDIVIDUAL ---
        with open(output_sm, "w", encoding="utf-8") as f:
            f.write(f"#TITLE:{titulo_simfile};\n#SUBTITLE:{subtitulo};\n")
            f.write(f"#ARTIST:{artist_name};\n#MUSIC:{new_audio_name};\n#BANNER:{new_banner_name};\n")
            f.write(f"#VIDEO:{new_video_name};\n")
            f.write(f"#OFFSET:-{val_offset:.3f};\n") 
            f.write(f"#BPMS:{bpms_string_line};\n")
            f.write(f"#BGCHANGES:{bg_changes_line};\n\n")
            
            for diff, bloque in mapa_pasos_por_dificultad.items():
                f.write(f"#NOTES:\n dance-single:\n AI_DSP_Hybrid_Engine:\n {diff}:\n {config_dificultad[diff]['meter']}:\n 0.1,0.1,0.1,0.1,0.1:\n")
                f.write(bloque)
                f.write("\n")

        chart_name = "DEATHSTREAM" if custom_params.get("max_notas_compas", 0) > 20 else "AI_Engine"

        # --- ESCRITURA DEL ARCHIVO .SSC INDIVIDUAL ---
        with open(output_ssc, "w", encoding="utf-8") as f:
            f.write(f"#VERSION:0.83;\n#TITLE:{titulo_simfile};\n#SUBTITLE:{subtitulo};\n")
            f.write(f"#ARTIST:{artist_name};\n#MUSIC:{new_audio_name};\n#BANNER:{new_banner_name};\n")
            f.write(f"#VIDEO:{new_video_name};\n")
            f.write(f"#OFFSET:-{val_offset:.3f};\n") 
            f.write(f"#BPMS:{bpms_string_line};\n#COMBOLINK:1;\n")

            if speeds_activo and speeds_string_line:
                f.write(f"#SPEEDS:{speeds_string_line};\n")

            f.write(f"#BGCHANGES:{bg_changes_line};\n\n")
            
            for diff, bloque in mapa_pasos_por_dificultad.items():
                f.write(f"//dance-single - AI_DSP_Hybrid_Engine\n#NOTEDATA:;\n#CHARTNAME:{chart_name};\n#STEPSTYPE:dance-single;\n")
                f.write(f"#DESCRIPTION:AI_DSP_Hybrid_Engine_Size_{int(round(pct*100))};\n#DIFFICULTY:{diff};\n#METER:{config_dificultad[diff]['meter']};\n")
                f.write(f"#RADARVALUES:0.1,0.1,0.1,0.1,0.1;\n#CREDIT:AI_Engine;\n#NOTES:\n")
                f.write(bloque)
                f.write("\n")

        if activar_muestreo:
            log_metricas_acumulado += f"📌 [VERSIÓN {int(round(pct*100))}%] Pasos procesados con éxito.\n"
        else:
            log_metricas_acumulado += log_metricas_diff

    # Estructura del log de cierre para la consola
    log_metricas_acumulado += f"\n⏱️ OFFSET GENERAL: {val_offset:.3f} s\n"
    log_metricas_acumulado += f"💓 BPM BASE GENERAL: {bpm:.3f} \n"
    log_metricas_acumulado += f"🔑 HUELLA DIGITAL (SEED): {seed_actual}\n"
    if activar_muestreo:
        log_metricas_acumulado += f"🚀 Proceso Multi-Capa completado: se exportaron {len(porcentajes_muestreo) * 2} archivos simfiles.\n"
    else:
        log_metricas_acumulado += f"🚀 Proceso Estándar completado: se exportaron 2 archivos simfiles.\n"
    log_metricas_acumulado += f"{'-'*45}\n"
        
    return bpm, duracion, log_metricas_acumulado

# =====================================================================
# 4. SUBVENTANA DEL VISUALIZADOR DE AUDIO INTERACTIVO (MATPLOTLIB)
# =====================================================================
class AudioVisualizerSubWindow(ctk.CTkToplevel):
    def __init__(self, master, audio_path, current_duration):
        super().__init__(master)
        self.master_app = master
        self.title("Límites de Audio Asimétricos")
        self.geometry("900x550")
        self.transient(master)  
        self.grab_set()  # Bloquea la interacción con la ventana base hasta cerrar
        
        self.markers_data = {}
        self.active_dragging_id = None
        self.background = None  
        self.last_update_time = 0

        # Cargar espectro rápido de audio
        self.y, self.sr = librosa.load(audio_path, sr=11025, mono=True)
        self.duration = float(current_duration if current_duration > 0 else librosa.get_duration(y=self.y, sr=self.sr))
        self.time_axis = np.linspace(0, self.duration, num=len(self.y))

        self.downsample_factor = max(1, len(self.y) // 8000)
        self.y_search = self.y[::self.downsample_factor]
        self.time_search = self.time_axis[::self.downsample_factor]

        lbl_info = ctk.CTkLabel(self, text="Arrastra las líneas: Offset (Izq) y Duración (Centro) frenan en el límite. Extensión (Der) puede expandirse.", font=ctk.CTkFont(size=12, slant="italic"))
        lbl_info.pack(pady=5)

        # Integrar Lienzo de Matplotlib
        self.fig = Figure(figsize=(7, 3.5), tight_layout=True, facecolor="#1e1e1e")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=ctk.BOTH, expand=True, padx=10, pady=5)

        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#151515")
        self.ax.plot(self.time_search, self.y_search, color='#2b5c8f', linewidth=1)
        self.ax.set_xlabel("Tiempo (s)", color="white")
        self.ax.set_ylabel("Amplitud", color="white")
        self.ax.tick_params(colors="white")
        
        self.ax.set_xlim(-5, self.duration + 20)
        self.ax.grid(True, alpha=0.2, color="gray")

        # Inicializar Marcadores Inteligentes
        init_pos = [self.duration * 0.15, self.duration * 0.80, self.duration * 0.95]
        self.create_full_marker("Offset", init_pos[0])
        self.create_full_marker("Duracion", init_pos[1])
        self.create_full_marker("Extension", init_pos[2])

        # Botón de sincronización
        btn_sync = ctk.CTkButton(self, text="Calcular y Sincronizar Valores", fg_color="#1abc9c", hover_color="#16a085", font=ctk.CTkFont(weight="bold"), command=self.procesar_y_enviar)
        btn_sync.pack(pady=10)

        # Conectar eventos de Matplotlib
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_drag)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('draw_event', self.on_draw_event)

    def create_full_marker(self, m_id, time_pos):
        line = self.ax.axvline(x=time_pos, color='red', linestyle='--', linewidth=1.5)
        point, = self.ax.plot(time_pos, 0.0, 'ro', markersize=8, markeredgecolor='white')
        
        bbox_props = dict(boxstyle="round,pad=0.2", fc="red", ec="white", lw=1, alpha=0.8)
        text = self.ax.text(time_pos, 0.0, f"{m_id}: {time_pos:.2f}s", color="white", ha="center", va="bottom", bbox=bbox_props)

        line.set_animated(True)
        point.set_animated(True)
        text.set_animated(True)
        self.markers_data[m_id] = {'line': line, 'point': point, 'text': text}

    def on_draw_event(self, event):
        if event is not None and event.canvas != self.canvas:
            return
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.draw_markers_manually()

    def draw_markers_manually(self):
        for marker in self.markers_data.values():
            self.ax.draw_artist(marker['line'])
            self.ax.draw_artist(marker['point'])
            self.ax.draw_artist(marker['text'])

    def on_click(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        click_threshold = (self.duration + 25) * 0.02
        for m_id, marker in self.markers_data.items():
            line_x = float(marker['line'].get_xdata()[0])
            if abs(event.xdata - line_x) < click_threshold:
                self.active_dragging_id = m_id
                break

    def on_drag(self, event):
        if self.active_dragging_id is None or event.inaxes != self.ax:
            return

        current_time = time.time()
        if current_time - self.last_update_time < 0.016: 
            return
        self.last_update_time = current_time

        max_limit = self.duration + 15 if self.active_dragging_id == "Extension" else self.duration
        new_x = max(0.0, min(event.xdata, max_limit))

        marker = self.markers_data[self.active_dragging_id]
        marker['line'].set_xdata([new_x, new_x])
        marker['point'].set_data([new_x], [0.0])
        marker['text'].set_position((new_x, 0.0))
        marker['text'].set_text(f"{self.active_dragging_id}: {new_x:.2f}s")

        if self.background is not None:
            self.canvas.restore_region(self.background)
            self.draw_markers_manually()
            self.canvas.blit(self.ax.bbox)

    def on_release(self, event):
        self.active_dragging_id = None

    def procesar_y_enviar(self):
        offset = float(self.markers_data["Offset"]['line'].get_xdata()[0])
        duracion = float(self.markers_data["Duracion"]['line'].get_xdata()[0])
        extension = float(self.markers_data["Extension"]['line'].get_xdata()[0])

        # Lógicas de filtrado condicional solicitadas
        val_offset = offset if offset < duracion else 0.0
        val_extension = (extension - duracion) if (extension - duracion) > 0 else 0.0

        # Aplicar límites máximos exigidos por los sliders nativos de CustomTkinter
        final_offset = min(16.0, val_offset)
        final_extension = min(15.0, val_extension)

        # Inyectar de regreso al panel maestro de CustomTkinter
        self.master_app.slider_offset.set(final_offset)
        self.master_app.actualizar_texto_offset(final_offset)
        
        self.master_app.entry_duracion.delete(0, "end")
        self.master_app.entry_duracion.insert(0, f"{duracion:.3f}")
        
        self.master_app.slider_extension.set(final_extension)
        self.master_app.actualizar_texto_extension(final_extension)

        messagebox.showinfo("Sincronización Exitosa", 
                            f"Valores ajustados con límites del motor:\n"
                            f"• Offset: {final_offset:.3f}s (Límite Máx 16s)\n"
                            f"• Duración: {duracion:.3f}s\n"
                            f"• Extensión: {final_extension:.1f}s (Límite Máx 15s)")
        self.destroy()

# =====================================================================
# 5. INTERFAZ GRÁFICA COMPATIBLE ADAPTATIVA CON SCROLL
# =====================================================================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class StepHybridUI(ctk.CTk):
    max_notas_compas = 12
    min_notas_compas = 8
    lineas_por_compas = 12

    def __init__(self):
        super().__init__()
        self.title("StepMania AI + DSP Dual Simfile Generator")
        self.geometry("480x640") # Ancho optimizado para distribución completamente vertical fija
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

        # =====================================================================
        # BLOQUE PRINCIPAL: DATOS DE ENTRADA Y ARCHIVOS ESENCIALES (FUERA)
        # =====================================================================
        self.btn_audio = ctk.CTkButton(self.contenedor_vertical, text="1. Seleccionar Canción (.mp3, .wav)", fg_color="#34495e", command=self.buscar_audio)
        self.label_audio_path = ctk.CTkLabel(self.contenedor_vertical, text="Ningún archivo seleccionado", text_color="gray", wraplength=350)

        self.btn_checkpoint = ctk.CTkButton(self.contenedor_vertical, text="2. Seleccionar Checkpoint IA (.pt)", fg_color="#2c3e50", command=self.buscar_checkpoint)
        self.label_checkpoint_path = ctk.CTkLabel(self.contenedor_vertical, text="Ningún modelo cargado", text_color="gray", wraplength=350)
        
        # Verificando si existe el checkpoint por default
        self.checkpoint_path = os.path.join(checkpoint_dir, model_name)
        if os.path.exists(self.checkpoint_path):
            print(f" 📦 Cargando pesos desde checkpoint histórico: {self.checkpoint_path}")
            self.checkpoint_file_path = self.checkpoint_path
            self.label_checkpoint_path.configure(text=f"{self.checkpoint_path}", text_color="gray")

        self.label_name = ctk.CTkLabel(self.contenedor_vertical, text="Título de la Canción:", font=ctk.CTkFont(weight="bold"))
        self.entry_title = ctk.CTkEntry(self.contenedor_vertical, placeholder_text="Ej: Cybernetic Beats", width=340)

        self.label_artist_name = ctk.CTkLabel(self.contenedor_vertical, text="Nombre del artista:", font=ctk.CTkFont(weight="bold"))
        self.entry_artist_name = ctk.CTkEntry(self.contenedor_vertical, placeholder_text="Ej: AI", width=340)

        #-- Meta datos --
        self.btn_banner = ctk.CTkButton(self.contenedor_vertical, text="Seleccionar Banner Graphic (Opcional)", fg_color="#16a085", command=self.buscar_banner)
        self.label_banner_path = ctk.CTkLabel(self.contenedor_vertical, text="Ningún banner seleccionado", text_color="gray", wraplength=350)
        self.btn_video = ctk.CTkButton(self.contenedor_vertical, text="Seleccionar Video de Fondo (Opcional)", fg_color="#8e44ad", command=self.buscar_video)
        self.label_video_path = ctk.CTkLabel(self.contenedor_vertical, text="Ningún video seleccionado", text_color="gray", wraplength=350)

        self.checkbox_rename = ctk.CTkCheckBox(self.contenedor_vertical, text="Renombrar archivos (Título de la Canción))") 

        # Separador visual lógico
        self.label_seccion_adv = ctk.CTkLabel(self.contenedor_vertical, text="--- Parámetros del Motor de Pasos (Originalidad) ---", font=ctk.CTkFont(size=13, weight="bold", slant="italic"), text_color="#95a5a6")
        self.label_temp = ctk.CTkLabel(self.contenedor_vertical, text="Temperatura IA (Caos): 1.30", font=ctk.CTkFont(weight="bold"))
        self.slider_temp = ctk.CTkSlider(self.contenedor_vertical, from_=0.5, to=1.5, number_of_steps=20, width=340, command=lambda v: self.label_temp.configure(text=f"Temperatura IA (Caos): {v:.2f}"))
        self.slider_temp.set(1.3)

        # Campo para el nombre del Song Pack
        self.label_pack_name = ctk.CTkLabel(self.contenedor_vertical, text="Nombre del Pack (Grupo):", font=ctk.CTkFont(weight="bold"))
        self.entry_pack_name = ctk.CTkEntry(self.contenedor_vertical, placeholder_text="Ej: Mi_AI_Pack_Vol1", width=340)
        self.entry_pack_name.insert(0, "AI_Generated_Charts") # Nombre por defecto

        self.label_seed = ctk.CTkLabel(self.contenedor_vertical, text="Semilla de Generación (Vacío = Aleatorio):", font=ctk.CTkFont(weight="bold"))
        self.entry_seed = ctk.CTkEntry(self.contenedor_vertical, placeholder_text="Ej: 12345 o texto_libre", width=340)

        # Empaquetado inmediato de los Datos de Entrada obligatorios
        componentes_entrada = [
            self.btn_audio, self.label_audio_path, self.btn_checkpoint, self.label_checkpoint_path,
            self.label_name, self.entry_title, self.label_artist_name, self.entry_artist_name, self.btn_banner, self.label_banner_path,
            self.btn_video, self.label_video_path, self.checkbox_rename, 
            self.label_seccion_adv, self.label_temp, self.slider_temp, 
            self.label_pack_name, self.entry_pack_name,
            self.label_seed, self.entry_seed
        ]
        for widget in componentes_entrada:
            widget.pack(pady=5, fill="x" if "Button" in type(widget).__name__ else None, padx=20)

        # =====================================================================
        # CONTROLADOR MAESTRO DE PRESETS DE PRUEBA
        # =====================================================================
        self.label_presets = ctk.CTkLabel(self.contenedor_vertical, text="🧪 Plantillas (Presets):", font=ctk.CTkFont(size=13, weight="bold"))
        self.label_presets.pack(pady=(15, 2), padx=20)

        self.menu_presets = ctk.CTkOptionMenu(
            self.contenedor_vertical,
            values=[
                "Seleccionar Preset (Manual)",
                "1. Visualmente dinámico.",
                "2. Más Saltos",
                "3. Velocidad caótica.",
                "4. Marea Flotante (Flujo de Olas y Smooth Scroll)",
                "5. Gimmick Caótico (Cortes Abruptos y Trampas de Impacto)",
                "6. Inferencia de Densidad Pura (Filtros Espectrales sin Modificadores)",
                "7. Tormenta Hardcore (Deathstream Máximo y Modificadores Coexistentes)"
            ],
            command=self.aplicar_preset_config,
            fg_color="#d35400",
            button_color="#e67e22"
        )
        self.menu_presets.pack(pady=5, padx=20, fill="x")

        # =====================================================================
        # SELECTOR DESPLEGABLE PARA APARTADOS CONFIGURACIÓN AVANZADA
        # =====================================================================
        self.label_menu_apartados = ctk.CTkLabel(self.contenedor_vertical, text="⚙️ Ajustes Avanzados del Motor:", font=ctk.CTkFont(size=13, weight="bold"))
        self.label_menu_apartados.pack(pady=(15, 2), padx=20)
        
        self.menu_apartados = ctk.CTkOptionMenu(
            self.contenedor_vertical, 
            values=["Ocultar Parámetros", "Settings de Tiempo", "Configuración de BPM y Ritmo", "Efectos, Minas y Trampas", "Filtros Espectrales y Dificultad"],
            command=self.conmutar_apartados_ui,
            fg_color="#2980b9",
            button_color="#3498db"
        )
        self.menu_apartados.pack(pady=5, padx=20, fill="x")

        # =====================================================================
        # CONTENEDORES DE APARTADOS (Frames ocultables)
        # =====================================================================
        self.apartado_bpm = ctk.CTkFrame(self.contenedor_vertical, fg_color="transparent")
        self.apartado_efectos = ctk.CTkFrame(self.contenedor_vertical, fg_color="transparent")
        self.apartado_filtros = ctk.CTkFrame(self.contenedor_vertical, fg_color="transparent")
        self.apartado_tiempo = ctk.CTkFrame(self.contenedor_vertical, fg_color="transparent")

        # --- APARTADO: TIEMPO ---
        self.btn_visualizar_grafico = ctk.CTkButton(self.apartado_tiempo, text="Ajustar Límites en Gráfica Interactiva 📊", fg_color="#8e44ad", hover_color="#9b59b6", font=ctk.CTkFont(weight="bold"), command=self.abrir_visualizador_audio)

        self.label_duracion = ctk.CTkLabel(self.apartado_tiempo, text="Duración Máxima (Segundos / 0=Full):", font=ctk.CTkFont(weight="bold"))
        self.entry_duracion = ctk.CTkEntry(self.apartado_tiempo, placeholder_text="Ej: 90", width=340)

        self.label_seccion_adv_tiempo = ctk.CTkLabel(self.apartado_tiempo, text="--- Parámetros Adicionales de la Canción ---", font=ctk.CTkFont(size=13, weight="bold", slant="italic"), text_color="#40FFEE")
        self.label_offset = ctk.CTkLabel(self.apartado_tiempo, text="Offset de Inicio: 0.000 s (Por defecto)", font=ctk.CTkFont(weight="bold"))
        self.slider_offset = ctk.CTkSlider(self.apartado_tiempo, from_=0.0, to=16.0, number_of_steps=400, width=340, command=self.actualizar_texto_offset)
        self.slider_offset.set(0.000)

        self.frame_offset_botones = ctk.CTkFrame(self.apartado_tiempo, fg_color="transparent")
        self.btn_offset_menos = ctk.CTkButton(self.frame_offset_botones, text="- 0.04s", width=80, fg_color="#7f8c8d", hover_color="#95a5a6", command=self.decrementar_offset)
        self.btn_offset_menos.pack(side="left", padx=10)
        self.btn_offset_mas = ctk.CTkButton(self.frame_offset_botones, text="+ 0.04s", width=80, fg_color="#7f8c8d", hover_color="#95a5a6", command=self.incrementar_offset)
        self.btn_offset_mas.pack(side="left", padx=10)

        self.checkbox_offset_auto = ctk.CTkCheckBox(
            self.apartado_tiempo, 
            text="Detectar Offset Automáticamente (DSP Vol)",
            text_color="#1abc9c",
            command=self.gestionar_exclusividad_offset
        )
        self.checkbox_offset_auto.select()

        # === Extensión final de la canción ===
        self.label_extension = ctk.CTkLabel(self.apartado_tiempo, text="Extensión Final Estética: 0.0 s (Corte normal)", font=ctk.CTkFont(weight="bold"))
        self.slider_extension = ctk.CTkSlider(self.apartado_tiempo, from_=0.0, to=15.0, number_of_steps=30, width=340, command=self.actualizar_texto_extension)
        self.slider_extension.set(0.0)

        widgets_tiempo = [
            self.btn_visualizar_grafico, self.label_duracion, self.entry_duracion,
            self.label_seccion_adv_tiempo, self.label_offset, self.slider_offset, self.frame_offset_botones, self.checkbox_offset_auto,
            self.label_extension, self.slider_extension
            ]
        for w in widgets_tiempo: 
            w.pack(pady=4, padx=20)

        # --- APARTADO: BPM Y RITMO ---
        self.label_seccion_adv_bpm = ctk.CTkLabel(self.apartado_bpm, text="--- Parámetros de BPM ---", font=ctk.CTkFont(size=13, weight="bold", slant="italic"), text_color="#A6408C")
        
        self.label_bpm = ctk.CTkLabel(self.apartado_bpm, text="Configuración de BPM: Auto (Detección DSP)", font=ctk.CTkFont(weight="bold"))
        #self.label_bpm.pack(pady=(10, 5), anchor="w")

        # Contenedor horizontal para el slider y el checkbox
        self.frame_controles_bpm = ctk.CTkFrame(self.apartado_bpm, fg_color="transparent")

        # Slider: Rango exacto de 60 a 300 con pasos de 0.5 (480 pasos)
        self.slider_bpm = ctk.CTkSlider(
            self.frame_controles_bpm, 
            from_=60, 
            to=300, 
            number_of_steps=480, 
            command=self.actualizar_texto_bpm, 
            width=260
        )
        self.slider_bpm.grid(row=0, column=0, padx=(0, 10), sticky="w")
        self.slider_bpm.set(60)

        # Checkbox para el modo Auto
        self.check_auto_bpm = ctk.CTkCheckBox(
            self.frame_controles_bpm, 
            text="Auto", 
            command=self.conmutar_auto_bpm,
            width=60
        )
        self.check_auto_bpm.grid(row=0, column=1, sticky="w")
        self.check_auto_bpm.select() # Inicia marcado por defecto (Auto activado)

        # Estado inicial: Desactivar slider porque "Auto" está marcado
        self.slider_bpm.configure(state="disabled")

        self.frame_bpm_botones = ctk.CTkFrame(self.apartado_bpm, fg_color="transparent")
        self.btn_bpm_menos = ctk.CTkButton(self.frame_bpm_botones, text="- 0.5", width=80, fg_color="#7f8c8d", hover_color="#95a5a6", command=self.decrementar_bpm)
        self.btn_bpm_menos.pack(side="left", padx=10)
        self.btn_bpm_menos.configure(state="disabled")
        self.btn_bpm_mas = ctk.CTkButton(self.frame_bpm_botones, text="+ 0.5", width=80, fg_color="#7f8c8d", hover_color="#95a5a6", command=self.incrementar_bpm)
        self.btn_bpm_mas.pack(side="left", padx=10)
        self.btn_bpm_mas.configure(state="disabled")

        self.checkbox_bpm = ctk.CTkCheckBox(self.apartado_bpm, text="Doble BPM (x2)") 

        self.checkbox_bpm_dinamico = ctk.CTkCheckBox(self.apartado_bpm, text="Aplicar BPM Dinámico (Alteraciones)", command=self.gestionar_exclusividad_ritmo)
        
        # Contenedor principal para la sección dinámica
        self.frame_bpm_dinamico = ctk.CTkFrame(self.apartado_bpm, fg_color="transparent")
        #self.frame_bpm_dinamico.pack(pady=10, fill="x")  # O usa .grid() según tu diseño

        # Fila 1: BPM Mínimo
        self.label_min_bpm_dinamico = ctk.CTkLabel(
            self.frame_bpm_dinamico, 
            text="BPM Mínimo (Campo vacío=Auto):", 
            width=160, 
            font=ctk.CTkFont(weight="bold"),
            anchor="w"
        )
        self.label_min_bpm_dinamico.pack(pady=2, padx=5)
        #self.label_min_bpm_dinamico.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.entry_min_bpm_dinamico = ctk.CTkEntry(
            self.frame_bpm_dinamico, 
            placeholder_text="Ej: 90", 
            width=100
        )
        self.entry_min_bpm_dinamico.pack(pady=2, padx=5)
        #self.entry_min_bpm_dinamico.grid(row=0, column=1, padx=5, pady=5)

        # Fila 2: BPM Máximo
        self.label_max_bpm_dinamico = ctk.CTkLabel(
            self.frame_bpm_dinamico, 
            text="BPM Máximo (Campo vacío=Auto):", 
            width=160, 
            font=ctk.CTkFont(weight="bold"),
            anchor="w"
        )
        self.label_max_bpm_dinamico.pack(pady=2, padx=5)
        #self.label_max_bpm_dinamico.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.entry_max_bpm_dinamico = ctk.CTkEntry(
            self.frame_bpm_dinamico, 
            placeholder_text="Ej: 140", 
            width=100
        )
        self.entry_max_bpm_dinamico.pack(pady=2, padx=5)
        #self.entry_max_bpm_dinamico.grid(row=1, column=1, padx=5, pady=5)

        # --- SLIDERS PROPIOS PARA EL CONTROL DE RMS EN BPM DINÁMICO ---
        # RMS Mínimo BPM
        self.label_rms_min_bpm = ctk.CTkLabel(self.frame_bpm_dinamico, text="Sensibilidad RMS Mínimo BPM: 0.50", font=ctk.CTkFont(weight="bold"), anchor="w")
        self.label_rms_min_bpm.pack(pady=2, padx=5)
        #self.label_rms_min_bpm.grid(row=2, column=0, columnspan=2, padx=5, pady=(10, 2), sticky="w")
        
        self.slider_rms_min_bpm = ctk.CTkSlider(self.frame_bpm_dinamico, from_=0.1, to=1.0, number_of_steps=18, width=320, 
                                                command=lambda v: self.label_rms_min_bpm.configure(text=f"Sensibilidad RMS Mínimo BPM: {v:.2f}"))
        self.slider_rms_min_bpm.pack(pady=2, padx=5)
        #self.slider_rms_min_bpm.grid(row=3, column=0, columnspan=2, padx=5, pady=5)
        self.slider_rms_min_bpm.set(0.50)

        # RMS Máximo BPM
        self.label_rms_max_bpm = ctk.CTkLabel(self.frame_bpm_dinamico, text="Sensibilidad RMS Máximo BPM: 1.50", font=ctk.CTkFont(weight="bold"), anchor="w")
        self.label_rms_max_bpm.pack(pady=2, padx=5)
        #self.label_rms_max_bpm.grid(row=4, column=0, columnspan=2, padx=5, pady=(10, 2), sticky="w")
        
        self.slider_rms_max_bpm = ctk.CTkSlider(self.frame_bpm_dinamico, from_=1.0, to=2.5, number_of_steps=30, width=320, 
                                                command=lambda v: self.label_rms_max_bpm.configure(text=f"Sensibilidad RMS Máximo BPM: {v:.2f}"))
        self.slider_rms_max_bpm.pack(pady=2, padx=5)
        #self.slider_rms_max_bpm.grid(row=5, column=0, columnspan=2, padx=5, pady=5)
        self.slider_rms_max_bpm.set(1.50)

        # Slider nuevo para la amortiguación del BPM
        self.label_bpm_amortiguador = ctk.CTkLabel(self.frame_bpm_dinamico, text="Amortiguador de Marea BPM: 0.12 (Atenuado)", font=ctk.CTkFont(weight="bold"), anchor="w")
        self.label_bpm_amortiguador.pack(pady=2, padx=5)
        
        self.slider_bpm_amortiguador = ctk.CTkSlider(
            self.frame_bpm_dinamico, from_=0.01, to=1.0, number_of_steps=99, width=320, 
            command=self.actualizar_texto_amortiguador_bpm
        )
        self.slider_bpm_amortiguador.pack(pady=2, padx=5)
        self.slider_bpm_amortiguador.set(0.12)

        self.label_seccion_adv_speed = ctk.CTkLabel(self.apartado_bpm, text="--- Parámetros de Scroll Speeds ---", font=ctk.CTkFont(size=13, weight="bold", slant="italic"), text_color="#67B3E6")
        self.checkbox_speeds_dinamico = ctk.CTkCheckBox(self.apartado_bpm, text="Adaptar Velocidad Visual (Scroll Speeds)", command=self.gestionar_exclusividad_ritmo)
        
        # CONTENEDOR DE SCROLL SPEED
        self.frame_scroll_speed_dinamico = ctk.CTkFrame(self.apartado_bpm, fg_color="transparent")
        self.label_speed_offset_time = ctk.CTkLabel(self.frame_scroll_speed_dinamico, text="Duración Extendida por Pérdida: 0.65% (Recomendado)", font=ctk.CTkFont(weight="bold"))
        self.label_speed_offset_time.pack(pady=2, padx=5)
        self.slider_speed_offset_time = ctk.CTkSlider(self.frame_scroll_speed_dinamico, from_=0, to=5, number_of_steps=500, width=340, command=lambda v: self.label_speed_offset_time.configure(text=f"Duración Extendida por Pérdida: {v:.2f}%" if v > 0 else "Duración Extendida por Pérdida: 0%"))
        self.slider_speed_offset_time.pack(pady=2, padx=5)
        self.slider_speed_offset_time.set(0.65)

        self.label_rms_min_speed = ctk.CTkLabel(self.frame_scroll_speed_dinamico, text="Sensibilidad RMS Mínimo Scroll: 0.50", font=ctk.CTkFont(weight="bold"))
        self.label_rms_min_speed.pack(pady=2, padx=5)
        self.slider_rms_min_speed = ctk.CTkSlider(self.frame_scroll_speed_dinamico, from_=0.1, to=1.0, number_of_steps=18, width=340, 
                                                 command=lambda v: self.label_rms_min_speed.configure(text=f"Sensibilidad RMS Mínimo Scroll: {v:.2f}"))
        self.slider_rms_min_speed.pack(pady=4, padx=5)
        self.slider_rms_min_speed.set(0.50)
        
        self.label_rms_max_speed = ctk.CTkLabel(self.frame_scroll_speed_dinamico, text="Sensibilidad RMS Máximo Scroll: 1.50", font=ctk.CTkFont(weight="bold"))
        self.label_rms_max_speed.pack(pady=2, padx=5)
        self.slider_rms_max_speed = ctk.CTkSlider(self.frame_scroll_speed_dinamico, from_=1.0, to=2.5, number_of_steps=30, width=340, 
                                                 command=lambda v: self.label_rms_max_speed.configure(text=f"Sensibilidad RMS Máximo Scroll: {v:.2f}"))
        self.slider_rms_max_speed.pack(pady=2, padx=5)
        self.slider_rms_max_speed.set(1.50)

        self.label_speed_min = ctk.CTkLabel(self.frame_scroll_speed_dinamico, text="Velocidad en Mínimos (Calma): 0.70x", font=ctk.CTkFont(weight="bold"))
        self.label_speed_min.pack(pady=2, padx=5)
        self.slider_speed_min = ctk.CTkSlider(self.frame_scroll_speed_dinamico, from_=0.25, to=1.0, number_of_steps=15, width=340, command=lambda v: self.label_speed_min.configure(text=f"Scroll Mínimo (Calma): {v:.2f}x"))
        self.slider_speed_min.pack(pady=2, padx=5)
        self.slider_speed_min.set(0.70)

        self.label_speed_max = ctk.CTkLabel(self.frame_scroll_speed_dinamico, text="Velocidad en Máximos (Drop): 1.40x", font=ctk.CTkFont(weight="bold"))
        self.label_speed_max.pack(pady=2, padx=5)
        self.slider_speed_max = ctk.CTkSlider(self.frame_scroll_speed_dinamico, from_=1.0, to=3.0, number_of_steps=40, width=340, command=lambda v: self.label_speed_max.configure(text=f"Scroll Máximo (Drop): {v:.2f}x"))
        self.slider_speed_max.pack(pady=2, padx=5)
        self.slider_speed_max.set(1.40)

        # Reemplazo de slider_speed_trans por slider_speed_trans
        self.label_speed_trans = ctk.CTkLabel(self.frame_scroll_speed_dinamico, text="Duración de Transición: 2.0 Beats (Suave)", font=ctk.CTkFont(weight="bold"))
        self.label_speed_trans.pack(pady=2, padx=5)
        self.slider_speed_trans = ctk.CTkSlider(self.frame_scroll_speed_dinamico, from_=0.0, to=4.0, number_of_steps=16, width=340, command=lambda v: self.label_speed_trans.configure(text=f"Duración de Transición: {v:.1f} Beats" if v > 0 else "Duración de Transición: Inmediata (0.0)"))
        self.slider_speed_trans.pack(pady=2, padx=5)
        self.slider_speed_trans.set(2.0)

        self.label_speed_umbral = ctk.CTkLabel(self.frame_scroll_speed_dinamico, text="Filtro Anti-Mareo (Umbral de Disparo): 0.50", font=ctk.CTkFont(weight="bold"))
        self.label_speed_umbral.pack(pady=2, padx=5)
        
        # Rango de 0.05 a 1.00 para cubrir todos tus escenarios probados
        self.slider_speed_umbral = ctk.CTkSlider(
            self.frame_scroll_speed_dinamico, from_=0.05, to=1.00, number_of_steps=19, width=340, 
            command=self.actualizar_texto_umbral_speed
        )
        self.slider_speed_umbral.pack(pady=2, padx=5)
        self.slider_speed_umbral.set(0.50)

        widgets_bpm = [
            self.label_seccion_adv_bpm, self.label_bpm, self.frame_controles_bpm, self.frame_bpm_botones, self.checkbox_bpm, self.checkbox_bpm_dinamico, #self.frame_bpm_dinamico,
            self.label_seccion_adv_speed, self.checkbox_speeds_dinamico
            ]
        for w in widgets_bpm: 
            w.pack(pady=4, padx=20)

        # --- APARTADO: MINAS Y TRAMPAS (FX) ---
        self.label_seccion_adv_trampas = ctk.CTkLabel(self.apartado_efectos, text="--- Parámetros de Efectos y Trampas ---", font=ctk.CTkFont(size=13, weight="bold", slant="italic"), text_color="#FF278B")

        self.label_prob_minas = ctk.CTkLabel(self.apartado_efectos, text="Probabilidad de Minas por compás: 35%", font=ctk.CTkFont(weight="bold"))
        self.slider_prob_minas = ctk.CTkSlider(self.apartado_efectos, from_=0, to=100, number_of_steps=100, width=340, command=lambda v: self.label_prob_minas.configure(text=f"Probabilidad de Minas por compás: {int(v)}%"))
        self.slider_prob_minas.set(35)
        
        self.label_max_minas = ctk.CTkLabel(self.apartado_efectos, text="Máximo Minas por Compás: 3", font=ctk.CTkFont(weight="bold"))
        self.slider_max_minas = ctk.CTkSlider(self.apartado_efectos, from_=0, to=4, number_of_steps=4, width=340, command=lambda v: self.label_max_minas.configure(text=f"Máximo Minas por Compás: {int(v)}"))
        self.slider_max_minas.set(3)

        self.label_prob_fakes = ctk.CTkLabel(self.apartado_efectos, text="Probabilidad de Fakes por compás: 25%", font=ctk.CTkFont(weight="bold"))
        self.slider_prob_fakes = ctk.CTkSlider(self.apartado_efectos, from_=0, to=100, number_of_steps=100, width=340, command=lambda v: self.label_prob_fakes.configure(text=f"Probabilidad de Fakes por compás: {int(v)}%"))
        self.slider_prob_fakes.set(25)

        self.label_max_fakes = ctk.CTkLabel(self.apartado_efectos, text="Máximo Fakes por Compás: 0", font=ctk.CTkFont(weight="bold"))
        self.slider_max_fakes = ctk.CTkSlider(self.apartado_efectos, from_=0, to=4, number_of_steps=4, width=340, command=lambda v: self.label_max_fakes.configure(text=f"Máximo Fakes por Compás: {int(v)}"))
        self.slider_max_fakes.set(0)

        self.label_prob_lifts = ctk.CTkLabel(self.apartado_efectos, text="Probabilidad de Lifts por compás: 25%", font=ctk.CTkFont(weight="bold"))
        self.slider_prob_lifts = ctk.CTkSlider(self.apartado_efectos, from_=0, to=100, number_of_steps=100, width=340, command=lambda v: self.label_prob_lifts.configure(text=f"Probabilidad de Lifts por compás: {int(v)}%"))
        self.slider_prob_lifts.set(25)

        self.label_max_lifts = ctk.CTkLabel(self.apartado_efectos, text="Máximo Lifts por Compás: 0", font=ctk.CTkFont(weight="bold"))
        self.slider_max_lifts = ctk.CTkSlider(self.apartado_efectos, from_=0, to=4, number_of_steps=4, width=340, command=lambda v: self.label_max_lifts.configure(text=f"Máximo Lifts por Compás: {int(v)}"))
        self.slider_max_lifts.set(0)

        self.label_prob_potions = ctk.CTkLabel(self.apartado_efectos, text="Probabilidad de Potions por compás: 15%", font=ctk.CTkFont(weight="bold"))
        self.slider_prob_potions = ctk.CTkSlider(self.apartado_efectos, from_=0, to=100, number_of_steps=100, width=340, command=lambda v: self.label_prob_potions.configure(text=f"Probabilidad de Potions por compás: {int(v)}%"))
        self.slider_prob_potions.set(15)

        self.label_max_potions = ctk.CTkLabel(self.apartado_efectos, text="Máximo Potions por Compás: 0", font=ctk.CTkFont(weight="bold"))
        self.slider_max_potions = ctk.CTkSlider(self.apartado_efectos, from_=0, to=4, number_of_steps=4, width=340, command=lambda v: self.label_max_potions.configure(text=f"Máximo Potions por Compás: {int(v)}"))
        self.slider_max_potions.set(0)

        self.label_prob_shields = ctk.CTkLabel(self.apartado_efectos, text="Probabilidad de Shields por compás: 15%", font=ctk.CTkFont(weight="bold"))
        self.slider_prob_shields = ctk.CTkSlider(self.apartado_efectos, from_=0, to=100, number_of_steps=100, width=340, command=lambda v: self.label_prob_shields.configure(text=f"Probabilidad de Shields por compás: {int(v)}%"))
        self.slider_prob_shields.set(15)

        self.label_max_shields = ctk.CTkLabel(self.apartado_efectos, text="Máximo Shields por Compás: 0", font=ctk.CTkFont(weight="bold"))
        self.slider_max_shields = ctk.CTkSlider(self.apartado_efectos, from_=0, to=4, number_of_steps=4, width=340, command=lambda v: self.label_max_shields.configure(text=f"Máximo Shields por Compás: {int(v)}"))
        self.slider_max_shields.set(0)

        self.label_prob_rayos = ctk.CTkLabel(self.apartado_efectos, text="Probabilidad de Rayos por compás: 15%", font=ctk.CTkFont(weight="bold"))
        self.slider_prob_rayos = ctk.CTkSlider(self.apartado_efectos, from_=0, to=100, number_of_steps=100, width=340, command=lambda v: self.label_prob_rayos.configure(text=f"Probabilidad de Rayos por compás: {int(v)}%"))
        self.slider_prob_rayos.set(15)

        self.label_max_rayos = ctk.CTkLabel(self.apartado_efectos, text="Máximo Rayos por Compás: 0", font=ctk.CTkFont(weight="bold"))
        self.slider_max_rayos = ctk.CTkSlider(self.apartado_efectos, from_=0, to=4, number_of_steps=4, width=340, command=lambda v: self.label_max_rayos.configure(text=f"Máximo Rayos por Compás: {int(v)}"))
        self.slider_max_rayos.set(0)

        self.label_prob_hiddens = ctk.CTkLabel(self.apartado_efectos, text="Probabilidad de Hiddens por compás: 20%", font=ctk.CTkFont(weight="bold"))
        self.slider_prob_hiddens = ctk.CTkSlider(self.apartado_efectos, from_=0, to=100, number_of_steps=100, width=340, command=lambda v: self.label_prob_hiddens.configure(text=f"Probabilidad de Hiddens por compás: {int(v)}%"))
        self.slider_prob_hiddens.set(20)

        self.label_max_hiddens = ctk.CTkLabel(self.apartado_efectos, text="Máximo Hiddens por Compás: 0", font=ctk.CTkFont(weight="bold"))
        self.slider_max_hiddens = ctk.CTkSlider(self.apartado_efectos, from_=0, to=4, number_of_steps=4, width=340, command=lambda v: self.label_max_hiddens.configure(text=f"Máximo Hiddens por Compás: {int(v)}"))
        self.slider_max_hiddens.set(0)

        self.checkbox_efectos_rms = ctk.CTkCheckBox(
            self.apartado_efectos, 
            text="Potenciar Efectos y Trampas en Drops (Análisis RMS)",
            text_color="#e67e22"
        )

        self.checkbox_efectos_rms.select()

        self.label_rms_min_fx = ctk.CTkLabel(self.apartado_efectos, text="Sensibilidad RMS Mínimo Trampas: 0.50", font=ctk.CTkFont(weight="bold"))
        self.slider_rms_min_fx = ctk.CTkSlider(self.apartado_efectos, from_=0.1, to=1.0, number_of_steps=18, width=340, 
                                              command=lambda v: self.label_rms_min_fx.configure(text=f"Sensibilidad RMS Mínimo Trampas: {v:.2f}"))
        self.slider_rms_min_fx.set(0.50)
        
        self.label_rms_max_fx = ctk.CTkLabel(self.apartado_efectos, text="Sensibilidad RMS Máximo Trampas: 1.50", font=ctk.CTkFont(weight="bold"))
        self.slider_rms_max_fx = ctk.CTkSlider(self.apartado_efectos, from_=1.0, to=2.5, number_of_steps=30, width=340, 
                                              command=lambda v: self.label_rms_max_fx.configure(text=f"Sensibilidad RMS Máximo Trampas: {v:.2f}"))
        self.slider_rms_max_fx.set(1.50)

        widgets_fx = [
            self.label_seccion_adv_trampas,
            self.label_prob_minas, self.slider_prob_minas, self.label_max_minas, self.slider_max_minas,
            self.label_prob_fakes, self.slider_prob_fakes, self.label_max_fakes, self.slider_max_fakes,
            self.label_prob_lifts, self.slider_prob_lifts, self.label_max_lifts, self.slider_max_lifts,
            self.label_prob_potions, self.slider_prob_potions, self.label_max_potions, self.slider_max_potions,
            self.label_prob_shields, self.slider_prob_shields, self.label_max_shields, self.slider_max_shields,
            self.label_prob_rayos, self.slider_prob_rayos, self.label_max_rayos, self.slider_max_rayos,
            self.label_prob_hiddens, self.slider_prob_hiddens, self.label_max_hiddens, self.slider_max_hiddens,
            self.checkbox_efectos_rms, self.label_rms_min_fx, self.slider_rms_min_fx, self.label_rms_max_fx, self.slider_rms_max_fx
            ]
        for w in widgets_fx:
            w.pack(pady=4, padx=20)

        # --- APARTADO: FILTROS ESPECTRALES, OFFSETS Y NOTAS ---
        self.label_seccion_adv_holders = ctk.CTkLabel(self.apartado_filtros, text="--- Parámetros de Holders ---", font=ctk.CTkFont(size=13, weight="bold", slant="italic"), text_color="#FFB874")

        self.label_max_hold = ctk.CTkLabel(self.apartado_filtros, text="Duración Máxima de Hold: 8 líneas", font=ctk.CTkFont(weight="bold"))
        self.slider_max_hold = ctk.CTkSlider(self.apartado_filtros, from_=2, to=32, number_of_steps=30, width=340, command=lambda v: self.label_max_hold.configure(text=f"Duración Máxima de Hold: {int(v)} líneas"))
        self.slider_max_hold.set(8)
        
        self.label_holds_sim = ctk.CTkLabel(self.apartado_filtros, text="Máximo de Holds simultáneos: 2", font=ctk.CTkFont(weight="bold"))
        self.slider_holds_sim = ctk.CTkSlider(self.apartado_filtros, from_=1, to=4, number_of_steps=3, width=340, command=lambda v: self.label_holds_sim.configure(text=f"Máximo de Holds simultáneos: {int(v)}"))
        self.slider_holds_sim.set(2)

        self.checkbox_postprocesar = ctk.CTkCheckBox(self.apartado_filtros, text="Aplicar Posprocesamiento Rítmico a los Holders", command=self.alternar_visibilidad_postprocesamiento)
        self.checkbox_postprocesar.select()

        self.checkbox_secciones_saltos = ctk.CTkCheckBox(
            self.apartado_filtros, 
            text="Generar Secciones de Saltos (Filtro RMS)",
            text_color="#9b59b6",
            command=self.gestionar_exclusividad_saltos
        )

        self.frame_saltos_dinamico = ctk.CTkFrame(self.apartado_filtros, fg_color="transparent")

        self.label_rms_min_saltos = ctk.CTkLabel(self.frame_saltos_dinamico, text="Sensibilidad RMS Mínimo Saltos: 0.50", font=ctk.CTkFont(weight="bold"))
        self.label_rms_min_saltos.pack(pady=2, padx=5)
        self.slider_rms_min_saltos = ctk.CTkSlider(self.frame_saltos_dinamico, from_=0.1, to=1.0, number_of_steps=18, width=340, 
                                                  command=lambda v: self.label_rms_min_saltos.configure(text=f"Sensibilidad RMS Mínimo Saltos: {v:.2f}"))
        self.slider_rms_min_saltos.pack(pady=2, padx=5)
        self.slider_rms_min_saltos.set(0.50)

        self.label_rms_max_saltos = ctk.CTkLabel(self.frame_saltos_dinamico, text="Sensibilidad RMS Máximo Saltos: 1.50", font=ctk.CTkFont(weight="bold"))
        self.label_rms_max_saltos.pack(pady=2, padx=5)
        self.slider_rms_max_saltos = ctk.CTkSlider(self.frame_saltos_dinamico, from_=1.0, to=2.5, number_of_steps=30, width=340, 
                                                  command=lambda v: self.label_rms_max_saltos.configure(text=f"Sensibilidad RMS Máximo Saltos: {v:.2f}"))
        self.slider_rms_max_saltos.pack(pady=2, padx=5)
        self.slider_rms_max_saltos.set(1.50)

        self.label_seccion_adv_dificultad = ctk.CTkLabel(self.apartado_filtros, text="--- Parámetros de Dificultad ---", font=ctk.CTkFont(size=13, weight="bold", slant="italic"), text_color="#4B00FF")
        
        self.label_nivel_texto = ctk.CTkLabel(self.apartado_filtros, text=f"Dificultad Techo del Pack: Nivel 16/30", font=ctk.CTkFont(size=13, weight="bold"), text_color="#3498db")
        self.slider_level = ctk.CTkSlider(self.apartado_filtros, from_=1, to=30, number_of_steps=29, width=340, command=self.actualizar_valores_interfaz)
        self.slider_level.set(16)

        self.checkbox_recalcular_diff = ctk.CTkCheckBox(self.apartado_filtros, text="Recalcular Dificultad Dinámicamente (NPS)", text_color="#3498db")
        self.checkbox_recalcular_diff.select() # Activado por defecto

        self.label_seccion_adv_compas = ctk.CTkLabel(self.apartado_filtros, text="--- Parámetros de Dificultad Adicional ---", font=ctk.CTkFont(size=13, weight="bold", slant="italic"), text_color="#C7FF00")

        self.label_rms_min = ctk.CTkLabel(self.apartado_filtros, text="Sensibilidad RMS Mínimo (Densidad de Notas): 0.50", font=ctk.CTkFont(weight="bold"))
        self.slider_rms_min = ctk.CTkSlider(self.apartado_filtros, from_=0.1, to=1.0, number_of_steps=18, width=340, command=lambda v: self.label_rms_min.configure(text=f"Sensibilidad RMS Mínimo (Calma): {v:.2f}"))
        self.slider_rms_min.set(0.5)
        
        self.label_rms_max = ctk.CTkLabel(self.apartado_filtros, text="Sensibilidad RMS Máximo (Densidad de Notas): 1.50", font=ctk.CTkFont(weight="bold"))
        self.slider_rms_max = ctk.CTkSlider(self.apartado_filtros, from_=1.0, to=2.5, number_of_steps=30, width=340, command=lambda v: self.label_rms_max.configure(text=f"Sensibilidad RMS Máximo (Drop): {v:.2f}"))
        self.slider_rms_max.set(1.5)

        self.label_lineas_por_compas_texto = ctk.CTkLabel(self.apartado_filtros, text=f"Lineas por compas: Precisión Estándar {self.lineas_por_compas}vas/192vas", font=ctk.CTkFont(size=13, weight="bold"), text_color="#16DB31")
        self.frame_lineas_por_compas_botones = ctk.CTkFrame(self.apartado_filtros, fg_color="transparent")
        self.btn_lineas_por_compas_menos_10 = ctk.CTkButton(self.frame_lineas_por_compas_botones, text="- 10", width=80, fg_color="#7f8c8d", hover_color="#95a5a6", command=self.decrementar_lineas_por_compas_ten)
        self.btn_lineas_por_compas_menos_10.pack(side="left", padx=10)
        self.btn_lineas_por_compas_menos = ctk.CTkButton(self.frame_lineas_por_compas_botones, text="- 1", width=80, fg_color="#7f8c8d", hover_color="#95a5a6", command=self.decrementar_lineas_por_compas_one)
        self.btn_lineas_por_compas_menos.pack(side="left", padx=10)
        self.btn_lineas_por_compas_mas = ctk.CTkButton(self.frame_lineas_por_compas_botones, text="+ 1", width=80, fg_color="#7f8c8d", hover_color="#95a5a6", command=self.incrementar_lineas_por_compas_one)
        self.btn_lineas_por_compas_mas.pack(side="left", padx=10)
        self.btn_lineas_por_compas_mas_10 = ctk.CTkButton(self.frame_lineas_por_compas_botones, text="+ 10", width=80, fg_color="#7f8c8d", hover_color="#95a5a6", command=self.incrementar_lineas_por_compas_ten)
        self.btn_lineas_por_compas_mas_10.pack(side="left", padx=10)

        self.label_min_notas_compas_texto = ctk.CTkLabel(self.apartado_filtros, text="MIN notas por compas: 8/192", font=ctk.CTkFont(size=13, weight="bold"), text_color="#16DB31")
        self.frame_min_notas_botones = ctk.CTkFrame(self.apartado_filtros, fg_color="transparent")
        self.btn_min_notas_menos_10 = ctk.CTkButton(self.frame_min_notas_botones, text="- 10", width=80, fg_color="#7f8c8d", hover_color="#95a5a6", command=self.decrementar_min_notas_compas_ten)
        self.btn_min_notas_menos_10.pack(side="left", padx=10)
        self.btn_min_notas_menos = ctk.CTkButton(self.frame_min_notas_botones, text="- 1", width=80, fg_color="#7f8c8d", hover_color="#95a5a6", command=self.decrementar_min_notas_compas_one)
        self.btn_min_notas_menos.pack(side="left", padx=10)
        self.btn_min_notas_mas = ctk.CTkButton(self.frame_min_notas_botones, text="+ 1", width=80, fg_color="#7f8c8d", hover_color="#95a5a6", command=self.incrementar_min_notas_compas_one)
        self.btn_min_notas_mas.pack(side="left", padx=10)
        self.btn_min_notas_mas_10 = ctk.CTkButton(self.frame_min_notas_botones, text="+ 10", width=80, fg_color="#7f8c8d", hover_color="#95a5a6", command=self.incrementar_min_notas_compas_ten)
        self.btn_min_notas_mas_10.pack(side="left", padx=10)

        self.label_max_notas_compas_texto = ctk.CTkLabel(self.apartado_filtros, text="MAX notas por compas: Dificultad Normal/Dificíl 12/192", font=ctk.CTkFont(size=13, weight="bold"), text_color="#16DB31")
        self.frame_max_notas_botones = ctk.CTkFrame(self.apartado_filtros, fg_color="transparent")
        self.btn_max_notas_menos_10 = ctk.CTkButton(self.frame_max_notas_botones, text="- 10", width=80, fg_color="#7f8c8d", hover_color="#95a5a6", command=self.decrementar_max_notas_compas_ten)
        self.btn_max_notas_menos_10.pack(side="left", padx=10)
        self.btn_max_notas_menos = ctk.CTkButton(self.frame_max_notas_botones, text="- 1", width=80, fg_color="#7f8c8d", hover_color="#95a5a6", command=self.decrementar_max_notas_compas_one)
        self.btn_max_notas_menos.pack(side="left", padx=10)
        self.btn_max_notas_mas = ctk.CTkButton(self.frame_max_notas_botones, text="+ 1", width=80, fg_color="#7f8c8d", hover_color="#95a5a6", command=self.incrementar_max_notas_compas_one)
        self.btn_max_notas_mas.pack(side="left", padx=10)
        self.btn_max_notas_mas_10 = ctk.CTkButton(self.frame_max_notas_botones, text="+ 10", width=80, fg_color="#7f8c8d", hover_color="#95a5a6", command=self.incrementar_max_notas_compas_ten)
        self.btn_max_notas_mas_10.pack(side="left", padx=10)

         # --- NUEVA SUBSECCIÓN: SISTEMA PARAMÉTRICO DE MUESTREO MULTI-ARCHIVO ---
        self.label_seccion_muestreo_adv = ctk.CTkLabel(self.apartado_filtros, text="--- Reducción Adaptativa de Mapas (Muestreo) ---", font=ctk.CTkFont(size=13, weight="bold", slant="italic"), text_color="#00FFCC")
        self.label_seccion_muestreo_adv.pack(pady=6, padx=20)

        # Checkbox muestreo por defecto DESACTIVADO
        self.checkbox_muestreo = ctk.CTkCheckBox(
            self.apartado_filtros, 
            text="Habilitar Generación de Muestras Multi-Capa", 
            text_color="#00FFCC",
            command=self.gestionar_exclusividad_muestreo
        )
        self.checkbox_muestreo.pack(pady=4, padx=20)
        self.checkbox_muestreo.deselect()

        # Contenedor dinámico secundario de Sliders
        self.frame_sub_muestreo = ctk.CTkFrame(self.apartado_filtros, fg_color="transparent")
        
        # Selección de porcentaje base (De 50% a 95%)
        self.label_muestreo_min = ctk.CTkLabel(self.frame_sub_muestreo, text="Muestreo Inicial Mínimo: 95%", font=ctk.CTkFont(weight="bold"))
        self.label_muestreo_min.pack(pady=2, padx=5)
        self.slider_muestreo_min = ctk.CTkSlider(self.frame_sub_muestreo, from_=0.50, to=0.95, number_of_steps=45, width=340, command=self.actualizar_texto_muestreo_min)
        self.slider_muestreo_min.pack(pady=2, padx=5)
        self.slider_muestreo_min.set(0.95)

        # Selección de muestras discretas (De 2 a 10)
        self.label_muestreo_num = ctk.CTkLabel(self.frame_sub_muestreo, text="Muestras Intermedias Totales: 6 (Hasta el 100%)", font=ctk.CTkFont(weight="bold"))
        self.label_muestreo_num.pack(pady=2, padx=5)
        self.slider_muestreo_num = ctk.CTkSlider(self.frame_sub_muestreo, from_=2, to=10, number_of_steps=8, width=340, command=self.actualizar_texto_muestreo_num)
        self.slider_muestreo_num.pack(pady=2, padx=5)
        self.slider_muestreo_num.set(6)

        widgets_filtros = [
            self.label_seccion_adv_holders,
            self.label_max_hold, self.slider_max_hold, self.label_holds_sim, self.slider_holds_sim, self.checkbox_postprocesar,
            self.checkbox_secciones_saltos,
            self.label_seccion_adv_dificultad,
            self.label_nivel_texto, self.slider_level, self.checkbox_recalcular_diff, 
            self.label_seccion_adv_compas,
            self.label_rms_min, self.slider_rms_min, self.label_rms_max, self.slider_rms_max,
            self.label_lineas_por_compas_texto, self.frame_lineas_por_compas_botones,
            self.label_min_notas_compas_texto, self.frame_min_notas_botones,
            self.label_max_notas_compas_texto, self.frame_max_notas_botones
            ]
        for w in widgets_filtros: 
            w.pack(pady=4, padx=20)

        #=====================================================================
        # BLOQUE FINAL E INMUTABLE DE EJECUCIÓN (FUERA)
        # =====================================================================

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
            self.btn_resetear, self.btn_generar, self.label_status, self.label_consola, self.txt_consola
            ]
        for widget in componentes_ui:
            widget.pack(pady=5, fill="x" if "Button" in type(widget).__name__ else None, padx=20)

    # --- NUEVO MÉTODO AUXILIAR PARA LA INTERMUTACIÓN DE APARTADOS ---
    def conmutar_apartados_ui(self, seleccion):
        """Muestra u oculta los sub-contenedores según la pestaña seleccionada del menú."""
        self.apartado_tiempo.pack_forget()
        self.apartado_bpm.pack_forget()
        self.apartado_efectos.pack_forget()
        self.apartado_filtros.pack_forget()
        if seleccion == "Settings de Tiempo":
            self.apartado_tiempo.pack(fill="x", expand=True, before=self.btn_resetear)
        elif seleccion == "Configuración de BPM y Ritmo":
            self.apartado_bpm.pack(fill="x", expand=True, before=self.btn_resetear)
        elif seleccion == "Efectos, Minas y Trampas":
            self.apartado_efectos.pack(fill="x", expand=True, before=self.btn_resetear)
        elif seleccion == "Filtros Espectrales y Dificultad":
            self.apartado_filtros.pack(fill="x", expand=True, before=self.btn_resetear)

    def aplicar_preset_config(self, seleccion):
        """
        Carga configuraciones automáticas calibradas con valores reales estables
        para evaluar el sistema de ráfagas de saltos y el presupuesto de trampas.
        """
        if seleccion == "Seleccionar Preset (Manual)":
            return

        # Limpieza obligatoria de buffers rítmicos dinámicos
        self.entry_min_bpm_dinamico.delete(0, "end")
        self.entry_max_bpm_dinamico.delete(0, "end")

        if seleccion == "1. Visualmente dinámico.":
            # --- PRIORIDAD: Activación balanceada de coexistencia rítmica ---
            self.checkbox_bpm_dinamico.select()
            self.checkbox_speeds_dinamico.select()
            self.checkbox_postprocesar.select()
            self.checkbox_secciones_saltos.deselect()
            self.checkbox_efectos_rms.select()
            
            self.slider_temp.set(1.10)
            self.slider_bpm_amortiguador.set(0.15)         # Reactivo progresivo estable
            self.slider_speed_trans.set(2.0)               # Rampa fluida de 2 compases
            self.slider_speed_umbral.set(0.50)              # Filtro anti-mareo óptimo
            self.slider_speed_min.set(0.80)
            self.slider_speed_max.set(1.35)
            
            # Trampas moderadas para no saturar el fondo dinámico
            self.slider_prob_minas.set(25)
            self.slider_max_minas.set(2)
            self.slider_prob_fakes.set(15)
            self.slider_max_fakes.set(1)

            self.slider_max_hold.set(4)

        elif seleccion == "2. Más Saltos":
            # --- PRIORIDAD: Liberación espectral de ráfagas (Jumpstreams) ---
            self.checkbox_bpm_dinamico.deselect()
            self.checkbox_speeds_dinamico.deselect()
            self.checkbox_postprocesar.select()
            self.checkbox_secciones_saltos.select()
            self.checkbox_efectos_rms.deselect()
            
            self.slider_temp.set(1.20)
            self.slider_rms_min_saltos.set(0.45)           # Más permisivo para capturar transitorios
            self.slider_rms_max_saltos.set(1.40)
            
            self.slider_max_hold.set(2)

            # Anular catálogo visual para evaluar puramente el flujo físico de flechas
            self.slider_max_minas.set(0)
            self.slider_max_fakes.set(0)
            self.slider_max_lifts.set(0)
            self.slider_max_hiddens.set(0)

        elif seleccion == "3. Velocidad caótica.":
            # --- PRIORIDAD: Gimmicks de scroll agresivos y cortes rígidos ---
            self.checkbox_bpm_dinamico.deselect()
            self.checkbox_speeds_dinamico.select()
            self.checkbox_postprocesar.select()
            self.checkbox_secciones_saltos.deselect()
            self.checkbox_efectos_rms.select()
            
            self.slider_temp.set(1.05)
            self.slider_speed_trans.set(0.0)               # Transiciones inmediatas (Cortes rígidos)
            self.slider_speed_umbral.set(0.25)              # Umbral sensible para disparos rápidos
            self.slider_speed_min.set(0.40)                # Frenados severos
            self.slider_speed_max.set(1.80)                # Aceleraciones súbitas
            
            # Minas estéticas espaciadas para marcar el peligro visual
            self.slider_prob_minas.set(30)
            self.slider_max_minas.set(1)

        elif seleccion == "4. Marea Flotante (Flujo de Olas y Smooth Scroll)":
            # --- ENFOQUE: Experiencia competitiva fluida sin fatiga visual ---
            self.checkbox_bpm_dinamico.select()
            self.checkbox_speeds_dinamico.select()
            self.checkbox_postprocesar.select()
            self.checkbox_secciones_saltos.deselect()
            self.checkbox_efectos_rms.select()
            
            self.slider_temp.set(1.00)
            self.slider_bpm_amortiguador.set(0.12)         # Flujo suavizado
            self.slider_speed_trans.set(3.0)               # Atenuación visual larga
            self.slider_speed_umbral.set(0.55)              # Estabilidad total
            self.slider_speed_min.set(0.85)
            self.slider_speed_max.set(1.25)
            
            # Presupuesto estricto de soporte decorativo
            self.slider_prob_minas.set(15)
            self.slider_max_minas.set(1)
            self.slider_max_hold.set(12)
            self.slider_holds_sim.set(2)

        elif seleccion == "5. Gimmick Caótico (Cortes Abruptos y Trampas de Impacto)":
            # --- ENFOQUE: Carta técnica avanzada con control de saturación ---
            self.checkbox_bpm_dinamico.select()
            self.checkbox_speeds_dinamico.select()
            self.checkbox_postprocesar.select()
            self.checkbox_secciones_saltos.select()
            self.checkbox_efectos_rms.select()
            
            self.slider_temp.set(1.35)
            self.slider_bpm_amortiguador.set(0.45)
            self.slider_speed_trans.set(0.5)               # Ataques rápidos tipo snap
            self.slider_speed_umbral.set(0.30)              # Reactivo controlado
            self.slider_speed_min.set(0.50)
            self.slider_speed_max.set(1.60)
            
            # Distribución balanceada de modificadores para no ahogar la pantalla
            self.slider_prob_minas.set(40)
            self.slider_max_minas.set(2)                    # Máximo 2 por compás
            self.slider_prob_fakes.set(25)
            self.slider_max_fakes.set(1)                    # Máximo 1 por compás
            self.slider_max_hold.set(6)
            self.slider_holds_sim.set(2)

        elif seleccion == "6. Inferencia de Densidad Pura (Filtros Espectrales sin Modificadores)":
            # --- ENFOQUE: Red Neuronal nativa pura sobre corrientes rítmicas ---
            self.checkbox_bpm_dinamico.deselect()
            self.checkbox_speeds_dinamico.deselect()
            self.checkbox_postprocesar.select()
            self.checkbox_secciones_saltos.select()
            self.checkbox_efectos_rms.deselect()
            
            self.slider_temp.set(1.15)
            self.slider_max_minas.set(0)
            self.slider_max_fakes.set(0)
            self.slider_max_lifts.set(0)
            self.slider_max_hiddens.set(0)
            
            self.slider_rms_min_saltos.set(0.50)
            self.slider_rms_max_saltos.set(1.50)
            self.slider_max_hold.set(8)
            self.slider_holds_sim.set(2)

        elif seleccion == "7. Tormenta Hardcore (Deathstream Máximo y Modificadores Coexistentes)":
            # --- ENFOQUE: Máxima subdivisión competitiva y ráfagas infinitas ---
            self.checkbox_bpm_dinamico.deselect()
            self.checkbox_speeds_dinamico.select()
            self.checkbox_postprocesar.select()
            self.checkbox_secciones_saltos.select()
            self.checkbox_efectos_rms.select()
            
            self.slider_temp.set(1.30)
            self.slider_speed_trans.set(1.5)               # Cambios espectrales rápidos pero legibles
            self.slider_speed_umbral.set(0.40)              # Sensibilidad media-alta
            self.slider_speed_min.set(0.90)
            self.slider_speed_max.set(1.45)
            
            # Escalar la subdivisión de la interfaz al modo experto
            self.lineas_por_compas = 16
            self.max_notas_compas = 16
            self.min_notas_compas = 8
            
            # Configuración intensa pero jugable de trampas gracias al presupuesto dinámico
            self.slider_prob_minas.set(20)
            self.slider_max_minas.set(2)
            self.slider_prob_fakes.set(15)
            self.slider_max_fakes.set(1)
            self.slider_max_hold.set(4)
            self.slider_holds_sim.set(2)

        # --- REFRESCO INMEDIATO DE LA INTERFAZ ---
        self.actualizar_texto_amortiguador_bpm(self.slider_bpm_amortiguador.get())
        self.label_speed_trans.configure(text=f"Duración de Transición: {self.slider_speed_trans.get():.1f} Beats" if self.slider_speed_trans.get() > 0 else "Duración de Transición: Inmediata (0.0)")
        self.actualizar_texto_umbral_speed(self.slider_speed_umbral.get())
        
        self.label_temp.configure(text=f"Temperatura IA (Caos): {self.slider_temp.get():.2f}")
        self.label_max_hold.configure(text=f"Duración Máxima de Hold: {int(self.slider_max_hold.get())} líneas")
        self.label_holds_sim.configure(text=f"Máximo de Holds simultáneos: {int(self.slider_holds_sim.get())}")
        self.label_prob_minas.configure(text=f"Probabilidad de Minas por compás: {int(self.slider_prob_minas.get())}%")
        self.label_max_minas.configure(text=f"Máximo Minas por Compás: {int(self.slider_max_minas.get())}")
        self.label_prob_fakes.configure(text=f"Probabilidad de Fakes por compás: {int(self.slider_prob_fakes.get())}%")
        self.label_max_fakes.configure(text=f"Máximo Fakes por Compás: {int(self.slider_max_fakes.get())}")
        
        # Sincronizar estados de visibilidad en los contenedores
        self.gestionar_exclusividad_ritmo()
        self.gestionar_exclusividad_saltos()
        self.actualizar_valores_compas()
        
        self.label_status.configure(text=f"Preset Cargado: {seleccion[3:]}", text_color="#1abc9c")
    
    def restablecer_valores(self):
        """ Devuelve todos los sliders avanzados a sus valores nativos por defecto """
        # --- LIMPIEZA DE RUTAS Y CAMPOS DE ARCHIVOS ---
        self.audio_file_path = ""
        #self.checkpoint_file_path = ""
        self.banner_file_path = ""
        self.video_file_path = ""
        
        self.label_audio_path.configure(text="Ningún archivo seleccionado", text_color="gray")
        self.label_banner_path.configure(text="Ningún banner seleccionado", text_color="gray")
        self.label_video_path.configure(text="Ningún video seleccionado", text_color="gray")
        
        self.entry_title.delete(0, "end")
        self.entry_artist_name.delete(0, "end")
        self.entry_duracion.delete(0, "end")
        self.entry_seed.delete(0, "end")

        # Resetear Sincronización Básica
        self.checkbox_rename.deselect()

        # Reset de ui de BPM
        self.slider_bpm.set(0)
        self.label_bpm.configure(text="Configuración de BPM: Auto (Detección DSP)")
        self.check_auto_bpm.select()
        self.slider_bpm.configure(state="disabled")
        self.btn_bpm_menos.configure(state="disabled")
        self.btn_bpm_mas.configure(state="disabled")

        self.checkbox_bpm.deselect() 

        self.checkbox_bpm_dinamico.deselect() 
        self.checkbox_bpm_dinamico.configure(state="normal")
        self.entry_min_bpm_dinamico.delete(0, "end")
        self.entry_max_bpm_dinamico.delete(0, "end")

        self.slider_rms_min_bpm.set(0.50)
        self.label_rms_min_bpm.configure(text="Sensibilidad RMS Mínimo BPM: 0.50")
        self.slider_rms_max_bpm.set(1.50)
        self.label_rms_max_bpm.configure(text="Sensibilidad RMS Máximo BPM: 1.50")

        self.label_bpm_amortiguador.configure("Amortiguador de Marea BPM: 0.12 (Atenuado)")
        self.slider_bpm_amortiguador.set(0.12)

        # Reset de ui de scrolls speeds
        self.checkbox_speeds_dinamico.deselect()
        self.checkbox_speeds_dinamico.configure(state="normal")

        self.label_speed_offset_time.configure(text="Duración Extendida por Pérdida: 0.65% (Recomendado)")
        self.slider_speed_offset_time.set(0.65)

        self.label_rms_min_speed.configure(text="Sensibilidad RMS Mínimo Scroll: 0.50")
        self.slider_rms_min_speed.set(0.50)
        self.label_rms_max_speed.configure(text="Sensibilidad RMS Máximo Scroll: 1.50")
        self.slider_rms_max_speed.set(1.50)

        self.slider_speed_min.set(0.70)
        self.slider_speed_min.configure(state="normal")
        self.label_speed_min.configure(text="Velocidad en Mínimos (Calma): 0.70x")
        self.slider_speed_max.set(1.40)
        self.slider_speed_max.configure(state="normal")
        self.label_speed_max.configure(text="Velocidad en Máximos (Drop): 1.40x")

        self.slider_speed_trans.set(2.0)
        self.slider_speed_trans.configure(state="normal")
        self.label_speed_trans.configure(text="Duración de Transición: 2.0 Beats (Suave)")

        self.slider_speed_umbral.set(0.50)
        self.label_speed_umbral.configure(text="Filtro Anti-Mareo (Umbral): 0.50 (Estable Óptimo)")

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

        self.label_rms_min_saltos.configure(text="Sensibilidad RMS Mínimo Saltos: 0.50")
        self.slider_rms_min_saltos.set(0.50)
        self.label_rms_max_saltos.configure(text="Sensibilidad RMS Máximo Saltos: 1.50")
        self.slider_rms_max_saltos.set(1.50)
        
        self.slider_rms_min.set(0.5)
        self.label_rms_min.configure(text="Sensibilidad RMS Mínimo (Densidad de Notas): 0.50")
        
        self.slider_rms_max.set(1.5)
        self.label_rms_max.configure(text="Sensibilidad RMS Máximo (Densidad de Notas): 1.50")
        
        # Resetear Techo de Dificultad
        self.slider_level.set(16)
        self.actualizar_valores_interfaz()
        
        #Resetear compas
        self.label_lineas_por_compas_texto.configure(text="Lineas por compas: Precisión Estándar 12vas/192vas", text_color="#16DB31")
        self.lineas_por_compas = 12
        self.label_max_notas_compas_texto.configure(text="MAX notas por compas: Dificultad Normal/Dificíl 12/192", text_color="#16DB31")
        self.max_notas_compas = 12
        self.label_min_notas_compas_texto.configure(text="MIN notas por compas: 8/192", text_color="#16DB31")
        self.min_notas_compas = 8

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

        self.label_rms_min_fx.configure(text="Sensibilidad RMS Mínimo Trampas: 0.50")
        self.slider_rms_min_fx.set(0.50)
        self.label_rms_max_fx.configure(text="Sensibilidad RMS Máximo Trampas: 1.50")
        self.slider_rms_max_fx.set(1.50)

        self.checkbox_muestreo.deselect()
        self.slider_muestreo_min.set(0.95)
        self.slider_muestreo_min.configure(state="disabled")
        self.label_muestreo_min.configure(text="Muestreo Inicial Mínimo: 95%")
        self.slider_muestreo_num.set(6)
        self.slider_muestreo_num.configure(state="disabled")
        self.label_muestreo_num.configure(text="Muestras Intermedias Totales: 6 (Hasta el 100%)")
        self.frame_sub_muestreo.pack_forget()

    def actualizar_texto_bpm(self, valor):
        # Redondeamos al 0.5 más cercano
        bpm = round(float(valor) * 2) / 2
        # Formatear para quitar el .0 si es un número entero
        bpm_texto = int(bpm) if bpm.is_integer() else bpm
        self.label_bpm.configure(text=f"Configuración de BPM: {bpm_texto} BPM (Manual)")

    def conmutar_auto_bpm(self):
        # Si el checkbox está marcado (Devuelve 1)
        if self.check_auto_bpm.get() == 1:
            self.slider_bpm.configure(state="disabled")
            self.btn_bpm_menos.configure(state="disabled")
            self.btn_bpm_mas.configure(state="disabled")
            self.label_bpm.configure(text="Configuración de BPM: Auto (Detección DSP)")
        else:
            self.slider_bpm.configure(state="normal")
            self.btn_bpm_menos.configure(state="normal")
            self.btn_bpm_mas.configure(state="normal")
            # Forzamos la actualización del texto con el valor actual del slider
            self.actualizar_texto_bpm(self.slider_bpm.get())
    
    def incrementar_bpm(self):
        # Incrementa en un paso del slider
        nuevo_valor = min(300, self.slider_bpm.get() + 0.5)
        self.slider_bpm.set(nuevo_valor)
        self.actualizar_texto_bpm(nuevo_valor)

    def decrementar_bpm(self):
        # Decrementa en un paso del slider (0 <- 300)
        nuevo_valor = max(60, self.slider_bpm.get() - 0.5)
        self.slider_bpm.set(nuevo_valor)
        self.actualizar_texto_bpm(nuevo_valor)
    
    def gestionar_exclusividad_ritmo(self):
        # Manejo independiente del panel de BPM Dinámico
        if self.checkbox_bpm_dinamico.get():
            self.frame_bpm_dinamico.pack(pady=5, fill="x", padx=20, after=self.checkbox_bpm_dinamico)
            self.slider_rms_min_bpm.configure(state="normal")
            self.slider_rms_max_bpm.configure(state="normal") 
            self.slider_bpm_amortiguador.configure(state="normal")
        else:
            self.frame_bpm_dinamico.pack_forget()

        # Manejo independiente del panel de Scroll Speeds
        if self.checkbox_speeds_dinamico.get():
            self.frame_scroll_speed_dinamico.pack(pady=5, fill="x", padx=20, after=self.checkbox_speeds_dinamico)
            self.slider_speed_min.configure(state="normal")
            self.slider_speed_max.configure(state="normal")
            self.slider_rms_max_speed.configure(state="normal")
            self.slider_rms_min_speed.configure(state="normal")
            self.slider_speed_trans.configure(state="normal")

        else:
            self.frame_scroll_speed_dinamico.pack_forget()

    def actualizar_texto_amortiguador_bpm(self, valor):
        if valor <= 0.05:
            tipo = "Extremo (Flujo de Olas)"
        elif valor <= 0.15:
            tipo = "Atenuado (Recomendado)"
        elif valor <= 0.40:
            tipo = "Reactivo Progresivo"
        else:
            tipo = "Inmediato (Brusco)"
        self.label_bpm_amortiguador.configure(text=f"Amortiguador de Marea BPM: {valor:.2f} ({tipo})")
    
    def actualizar_texto_umbral_speed(self, valor):
        # Redondeamos a dos decimales para mantener limpia la UI
        valor = round(float(valor), 2)
        if valor >= 0.50:
            tipo = "Estable Óptimo (Cero Mareos)"
        elif valor >= 0.30:
            tipo = "Sensible (Riesgo de Mareo)"
        elif valor >= 0.20:
            tipo = "Cambios Bruscos / Gimmick"
        else:
            tipo = "Hiper-Reactivo (Inestable)"
        self.label_speed_umbral.configure(text=f"Filtro Anti-Mareo (Umbral): {valor:.2f} ({tipo})")

    def gestionar_exclusividad_saltos(self):
        if self.checkbox_secciones_saltos.get():
            self.frame_saltos_dinamico.pack(pady=5, fill="x", padx=20, after=self.checkbox_secciones_saltos)
        else:
            self.frame_saltos_dinamico.forget()

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
    
    #-----------------Muestreo-------------------------
    def gestionar_exclusividad_muestreo(self):
        """Muestra u oculta los sub-controles de muestreo de acuerdo al estado del checkbox principal."""
        if self.checkbox_muestreo.get():
            self.frame_sub_muestreo.pack(pady=5, fill="x", padx=20, after=self.checkbox_muestreo)
            self.slider_muestreo_min.configure(state="normal")
            self.slider_muestreo_num.configure(state="normal")
        else:
            self.frame_sub_muestreo.pack_forget()

    def actualizar_texto_muestreo_min(self, valor):
        pct = int(float(valor) * 100)
        self.label_muestreo_min.configure(text=f"Muestreo Inicial Mínimo: {pct}%")

    def actualizar_texto_muestreo_num(self, valor):
        muestras = int(valor)
        self.label_muestreo_num.configure(text=f"Muestras Intermedias Totales: {muestras} (Hasta el 100%)")
    #----------------------------------------------------
    
    def actualizar_texto_extension(self, valor):
        self.label_extension.configure(text=f"Extensión Final Estética: {valor:.1f} s")

    def actualizar_valores_interfaz(self, *args):
        level = int(self.slider_level.get())
        e = max(1, int(level * 0.25))
        m = max(2, int(level * 0.50))
        h = max(3, int(level * 0.75))
        self.label_nivel_texto.configure(text=f"Dificultad Techo: Nivel {level}\nEscala: [Easy {e} | Med {m} | Hard {h} | Chal {level}]")

    def incrementar_lineas_por_compas_one(self):
        self.lineas_por_compas = min(192, self.lineas_por_compas + 1)
        self.actualizar_valores_compas()

    def incrementar_lineas_por_compas_ten(self):
        self.lineas_por_compas = min(192, self.lineas_por_compas + 10)
        self.actualizar_valores_compas()

    def decrementar_lineas_por_compas_one(self):
        self.lineas_por_compas = max(2, self.lineas_por_compas - 1)
        self.actualizar_valores_compas()

    def decrementar_lineas_por_compas_ten(self):
        self.lineas_por_compas = max(2, self.lineas_por_compas - 10)
        self.actualizar_valores_compas()

    def incrementar_max_notas_compas_one(self):
        self.max_notas_compas = min(192, self.max_notas_compas + 1)
        self.actualizar_valores_compas()

    def incrementar_max_notas_compas_ten(self):
        self.max_notas_compas = min(192, self.max_notas_compas + 10)
        self.actualizar_valores_compas()

    def decrementar_max_notas_compas_one(self):
        self.max_notas_compas = max(2, self.max_notas_compas - 1)
        self.actualizar_valores_compas()

    def decrementar_max_notas_compas_ten(self):
        self.max_notas_compas = max(2, self.max_notas_compas - 10)
        self.actualizar_valores_compas()
    
    def incrementar_min_notas_compas_one(self):
        self.min_notas_compas = min(192, self.min_notas_compas + 1)
        self.actualizar_valores_compas()

    def incrementar_min_notas_compas_ten(self):
        self.min_notas_compas = min(192, self.min_notas_compas + 10)
        self.actualizar_valores_compas()

    def decrementar_min_notas_compas_one(self):
        self.min_notas_compas = max(2, self.min_notas_compas - 1)
        self.actualizar_valores_compas()

    def decrementar_min_notas_compas_ten(self):
        self.min_notas_compas = max(2, self.min_notas_compas - 10)
        self.actualizar_valores_compas()

    def actualizar_valores_compas(self):
        
        if(self.min_notas_compas > self.lineas_por_compas):
            self.min_notas_compas = 2
        if(self.max_notas_compas > self.lineas_por_compas):
            self.max_notas_compas = self.lineas_por_compas

        if(self.lineas_por_compas < 12):
            self.label_lineas_por_compas_texto.configure(text=f"Lineas por compas: Precisión Simple {self.lineas_por_compas}vas/192vas", text_color="#11CCDB")
            #Corrección de congruencia entre mínimo y máximo de notas
            if(self.min_notas_compas >= 12 or self.max_notas_compas >= 12 or self.max_notas_compas < self.min_notas_compas):
                self.min_notas_compas = 2
                self.max_notas_compas = self.lineas_por_compas
                self.label_min_notas_compas_texto.configure(text=f"MIN notas por compas: {self.min_notas_compas}/192", text_color="#11CCDB")
                self.label_max_notas_compas_texto.configure(text=f"MAX notas por compas: Dificultad Baja/Normal {self.max_notas_compas}/192", text_color="#11CCDB")
            else:
                self.label_min_notas_compas_texto.configure(text=f"MIN notas por compas: {self.min_notas_compas}/192", text_color="#11CCDB")
                self.label_max_notas_compas_texto.configure(text=f"MAX notas por compas: Dificultad Baja/Normal {self.max_notas_compas}/192", text_color="#11CCDB")
        elif(self.lineas_por_compas <= 16):
            self.label_lineas_por_compas_texto.configure(text=f"Lineas por compas: Precisión Estándar {self.lineas_por_compas}vas/192vas", text_color="#16DB31")
            #Corrección de congruencia entre mínimo y máximo de notas
            if(self.min_notas_compas > 16 or self.max_notas_compas > 16 or self.max_notas_compas < self.min_notas_compas):
                self.min_notas_compas = 2
                self.max_notas_compas = self.lineas_por_compas
                self.label_min_notas_compas_texto.configure(text=f"MIN notas por compas: {self.min_notas_compas}/192", text_color="#16DB31")
                self.label_max_notas_compas_texto.configure(text=f"MAX notas por compas: Dificultad Normal/Dificíl {self.max_notas_compas}/192", text_color="#16DB31")
            else:
                self.label_min_notas_compas_texto.configure(text=f"MIN notas por compas: {self.min_notas_compas}/192", text_color="#16DB31")
                self.label_max_notas_compas_texto.configure(text=f"MAX notas por compas: Dificultad Normal/Dificíl {self.max_notas_compas}/192", text_color="#16DB31")
        elif(self.lineas_por_compas <= 64):
            self.label_lineas_por_compas_texto.configure(text=f"Lineas por compas: Precisión Alta {self.lineas_por_compas}vas/192vas", text_color="#C600DB")
            #Corrección de congruencia entre mínimo y máximo de notas
            if(self.min_notas_compas > 64 or self.max_notas_compas > 64 or self.max_notas_compas < self.min_notas_compas):
                self.min_notas_compas = 2
                self.max_notas_compas = self.lineas_por_compas
                self.label_min_notas_compas_texto.configure(text=f"MIN notas por compas: {self.min_notas_compas}/192", text_color="#FF9C00")
                self.label_max_notas_compas_texto.configure(text=f"MAX notas por compas: Dificultad Dificíl/Experto {self.max_notas_compas}/192", text_color="#FF9C00")
            else:
                self.label_min_notas_compas_texto.configure(text=f"MIN notas por compas: {self.min_notas_compas}/192", text_color="#FF9C00")
                self.label_max_notas_compas_texto.configure(text=f"MAX notas por compas: Dificultad Dificíl/Experto {self.max_notas_compas}/192", text_color="#FF9C00")
        else:
            self.label_lineas_por_compas_texto.configure(text=f"Lineas por compas: Precisión Milimétrica {self.lineas_por_compas}vas/192vas", text_color="#FF7200")
            #Corrección de congruencia entre mínimo y máximo de notas
            if(self.max_notas_compas < self.min_notas_compas):
                self.min_notas_compas = 2
                self.max_notas_compas = self.lineas_por_compas
                self.label_min_notas_compas_texto.configure(text=f"MIN notas por compas: {self.min_notas_compas}/192", text_color="#FF0000")
                self.label_max_notas_compas_texto.configure(text=f"MAX notas por compas: Dificultad Experto/Máquina {self.max_notas_compas}/192", text_color="#FF0000")
            else:
                self.label_min_notas_compas_texto.configure(text=f"MIN notas por compas: {self.min_notas_compas}/192", text_color="#FF0000")
                self.label_max_notas_compas_texto.configure(text=f"MAX notas por compas: Dificultad Experto/Máquina {self.max_notas_compas}/192", text_color="#FF0000")
        
    def buscar_audio(self):
        file_path = filedialog.askopenfilename(filetypes=[("Archivos de Audio", "*.mp3 *.wav *.ogg *.flac")])
        if file_path:
            self.audio_file_path = file_path
            self.label_audio_path.configure(text=os.path.basename(file_path), text_color="#1abc9c")
            self.entry_title.delete(0, "end")
            self.entry_title.insert(0, os.path.splitext(os.path.basename(file_path))[0])

    def abrir_visualizador_audio(self):
        """Abre la subventana gráfica interactiva de Matplotlib usando Blitting estático."""
        if not self.audio_file_path:
            messagebox.showwarning("Falta Archivo", "Por favor selecciona primero un archivo de audio válido en el paso 1.")
            return
            
        duracion_manual = 0.0
        dur_raw = self.entry_duracion.get().strip()
        if dur_raw:
            try:
                duracion_manual = float(dur_raw)
            except ValueError:
                pass
                
        # Crear subventana pasándole la app maestra (self)
        AudioVisualizerSubWindow(self, self.audio_file_path, duracion_manual)

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

        min_bpm_dinamico_text = self.entry_min_bpm_dinamico.get().strip()
        max_bpm_dinamico_text = self.entry_max_bpm_dinamico.get().strip()
        min_bpm_dinamico = 0
        max_bpm_dinamico = 0
        if min_bpm_dinamico_text and max_bpm_dinamico_text:
            try:
                min_bpm_dinamico = int(min_bpm_dinamico_text)
                max_bpm_dinamico = int(max_bpm_dinamico_text)
                if min_bpm_dinamico < 30 or max_bpm_dinamico > 300 or max_bpm_dinamico == 0 or min_bpm_dinamico == 0: raise ValueError
            except ValueError:
                messagebox.showerror("Error", "El BPM del mínimo o máximo no es válido, no puede ser superior a 300 o menor a 30") 
                return

        params_usuario = {
            "renombrar_archivos": bool(self.checkbox_rename.get()),
            "bpm_manual": float(self.slider_bpm.get()),
            "bpm_automatico": bool(self.check_auto_bpm.get()),
            "double_bpm": bool(self.checkbox_bpm.get()),
            "aplicar_bpm_dinamico": bool(self.checkbox_bpm_dinamico.get()), 
            "min_bpm_dinamico" : min_bpm_dinamico,
            "max_bpm_dinamico" : max_bpm_dinamico,
            "bpm_dinamico_rms_min": float(self.slider_rms_min_bpm.get()),
            "bpm_dinamico_rms_max": float(self.slider_rms_max_bpm.get()),
            "bpm_amortiguador_custom": float(self.slider_bpm_amortiguador.get()), # Marea BPM
            "aplicar_speeds_dinamicos": bool(self.checkbox_speeds_dinamico.get()),
            "speed_offset_time": float(self.slider_speed_offset_time.get()),
            "speed_rms_min": float(self.slider_rms_min_speed.get()),
            "speed_rms_max": float(self.slider_rms_max_speed.get()),
            "speed_min_custom": float(self.slider_speed_min.get()),  
            "speed_max_custom": float(self.slider_speed_max.get()),  
            "speed_trans_custom": float(self.slider_speed_trans.get()), # Marea Scroll
            "speed_umbral_disparo": float(self.slider_speed_umbral.get()),
            "aplicar_postprocesamiento": bool(self.checkbox_postprocesar.get()),
            "recalcular_dificultad": bool(self.checkbox_recalcular_diff.get()),
            "offset_automatico": bool(self.checkbox_offset_auto.get()),
            "offset_manual": float(self.slider_offset.get()),
            "extension_final": float(self.slider_extension.get()),
            "temperatura": float(self.slider_temp.get()),
            "max_hold": int(self.slider_max_hold.get()),
            "holds_simultaneos": int(self.slider_holds_sim.get()),
            "activar_secciones_saltos": bool(self.checkbox_secciones_saltos.get()),
            "saltos_rms_min": float(self.slider_rms_min_saltos.get()),
            "saltos_rms_max": float(self.slider_rms_max_saltos.get()),
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
            "fx_rms_min": float(self.slider_rms_min_fx.get()),
            "fx_rms_max": float(self.slider_rms_max_fx.get()),
            "ratio_min_energia": float(self.slider_rms_min.get()),
            "ratio_max_energia": float(self.slider_rms_max.get()),
            "lineas_por_compas": int(self.lineas_por_compas),
            "max_notas_compas": int(self.max_notas_compas),
            "min_notas_compas": int(self.min_notas_compas),
            "pack_name": self.entry_pack_name.get().strip() if self.entry_pack_name.get().strip() else "AI_Generated_Charts",
            "activar_muestreo_multicapa": bool(self.checkbox_muestreo.get()),
            "muestreo_pct_min": float(self.slider_muestreo_min.get()),
            "muestreo_num_muestras": int(self.slider_muestreo_num.get()),
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
        
        #carpeta_salida = os.path.dirname(self.audio_file_path)
        # PARCHE: Determinar de forma dinámica el directorio exacto donde se ejecuta este script
        import sys
        if getattr(sys, 'frozen', False):
            # Si el script está compilado con PyInstaller (.exe)
            carpeta_salida = os.path.dirname(sys.executable)
        else:
            # Si el script se está corriendo nativamente desde la consola (.py)
            carpeta_salida = os.path.dirname(os.path.abspath(__file__))
           

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
