import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import io
import base64
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(page_title="Análisis de Señales y Sistemas", layout="wide")

# Título principal
st.title(" Análisis de Señales y Sistemas")
st.markdown("### Laboratorio 4: Series y transformada de Fourier")

# Sidebar para navegación
st.sidebar.title(" Navegación")
pagina = st.sidebar.radio(
    "Seleccione el ejercicio:",
    ["Punto 1: Series de Fourier", "Punto 2: Modulación AM", "Punto 3: Modulación y Demodulación en cuadratura de fase", "Punto 4: Modulación de amplitud DSB-LC"]
)

st.sidebar.markdown("---")

# ============================================================================
# PUNTO 1: SERIES DE FOURIER
# ============================================================================

if pagina == "Punto 1: Series de Fourier":
    st.header("📊 Punto 1: Análisis de Series de Fourier")
    st.markdown("**Representación gráfica de coeficientes y reconstrucción de señales periódicas**")
    
    # Definición de las señales
    def senal_triangular(t, T):
        """Señal triangular periódica (Ejemplo 3.6.1)"""
        t_mod = np.mod(t + T/2, T) - T/2
        return np.where(t_mod < 0, 1 + 4*t_mod/T, 1 - 4*t_mod/T)

    def senal_rampa(t, T):
        """Señal rampa periódica (Ejemplo 3.6.2)"""
        t_mod = np.mod(t + T/2, T) - T/2
        return t_mod

    def senal_cuadratica(t, T):
        """Función cuadrática periódica (Ejemplo 3.6.3)"""
        t_mod = np.mod(t + T/2, T) - T/2
        return t_mod**2

    def senal_mixta(t, T):
        """Función definida en [-1, 1] periódica (Ejemplo 3.6.4)"""
        result = np.zeros_like(t)
        t_mod = np.mod(t + T/2, T) - T/2
        
        for i, t_val in enumerate(t_mod):
            if -1 <= t_val < 0:
                result[i] = t_val
            elif 0 <= t_val < 1:
                result[i] = 1
            else:
                t_base = np.mod(t_val + 1, 2) - 1
                if -1 <= t_base < 0:
                    result[i] = t_base
                elif 0 <= t_base <= 1:
                    result[i] = 1
        
        return result

    # Coeficientes analíticos según el libro
    def coeficientes_triangular(N):
        c_n = [0]
        n_values = [0]
        an_list = []
        bn_list = []
        
        for n in range(1, N + 1):
            an = (4 / (n**2 * np.pi**2)) * (1 - np.cos(n * np.pi))
            bn = 0
            an_list.append(an)
            bn_list.append(bn)
            c_n.append(an)
            n_values.append(n)
        
        return np.array(c_n), np.array(n_values), 0, np.array(an_list), np.array(bn_list)

    def coeficientes_rampa(N):
        c_n = [0]
        n_values = [0]
        an_list = []
        bn_list = []
        
        for n in range(1, N + 1):
            an = 0
            bn = (-2 / n) * np.cos(n * np.pi)
            an_list.append(an)
            bn_list.append(bn)
            c_n.append(bn)
            n_values.append(n)
        
        return np.array(c_n), np.array(n_values), 0, np.array(an_list), np.array(bn_list)

    def coeficientes_cuadratica(N):
        a0 = np.pi**2 / 3
        c_n = [a0]
        n_values = [0]
        an_list = []
        bn_list = []
        
        for n in range(1, N + 1):
            an = (4 / n**2) * np.cos(n * np.pi)
            bn = 0
            an_list.append(an)
            bn_list.append(bn)
            c_n.append(an)
            n_values.append(n)
        
        return np.array(c_n), np.array(n_values), a0, np.array(an_list), np.array(bn_list)

    def coeficientes_mixta(N):
        a0 = 1/4
        c_n = [a0]
        n_values = [0]
        an_list = []
        bn_list = []
        
        for n in range(1, N + 1):
            if n % 2 == 0:
                an = 0
            else:
                an = 2 / (n**2 * np.pi**2)
            
            if n % 2 == 0:
                bn = -1 / (n * np.pi)
            else:
                bn = 3 / (n * np.pi)
            
            an_list.append(an)
            bn_list.append(bn)
            cn = np.sqrt(an**2 + bn**2)
            c_n.append(cn)
            n_values.append(n)
        
        return np.array(c_n), np.array(n_values), a0, np.array(an_list), np.array(bn_list)

    def reconstruir_senal(an_list, bn_list, t, a0, usar_pi=False):
        y = a0 * np.ones_like(t)
        for n in range(len(an_list)):
            if usar_pi:
                y += an_list[n] * np.cos((n+1) * np.pi * t) + bn_list[n] * np.sin((n+1) * np.pi * t)
            else:
                y += an_list[n] * np.cos((n+1) * t) + bn_list[n] * np.sin((n+1) * t)
        return y

    # Configuración
    st.sidebar.header("⚙️ Configuración")
    
    tipo_senal = st.sidebar.selectbox(
        "Seleccione la señal:",
        ["Señal triangular (Ej. 3.6.1)", 
         "Señal rampa (Ej. 3.6.2)", 
         "Función cuadrática (Ej. 3.6.3)",
         "Función definida en [-1,1] (Ej. 3.6.4)"]
    )

    if tipo_senal == "Función definida en [-1,1] (Ej. 3.6.4)":
        T = 2
        st.sidebar.info(f"Período fijo T = {T}")
    else:
        T = 2 * np.pi
        st.sidebar.info(f"Período T = 2π (fórmulas del libro)")

    if tipo_senal == "Señal triangular (Ej. 3.6.1)":
        senal_func = senal_triangular
        coef_func = coeficientes_triangular
    elif tipo_senal == "Señal rampa (Ej. 3.6.2)":
        senal_func = senal_rampa
        coef_func = coeficientes_rampa
    elif tipo_senal == "Función cuadrática (Ej. 3.6.3)":
        senal_func = senal_cuadratica
        coef_func = coeficientes_cuadratica
    else:
        senal_func = senal_mixta
        coef_func = coeficientes_mixta

    # Señal Original
    st.subheader("📈 Señal Original")
    fig_original, ax_original = plt.subplots(figsize=(14, 5))
    t_plot = np.linspace(-T, T, 2000)
    y_original = senal_func(t_plot, T)
    ax_original.plot(t_plot, y_original, 'b-', linewidth=2.5)
    ax_original.set_xlabel('Tiempo (t)', fontsize=12)
    ax_original.set_ylabel('x(t)', fontsize=12)
    ax_original.set_title(f'{tipo_senal}', fontsize=14, fontweight='bold')
    ax_original.grid(True, alpha=0.3)
    ax_original.axhline(y=0, color='k', linewidth=0.5)
    ax_original.axvline(x=0, color='k', linewidth=0.5)
    ax_original.axvline(x=-T, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    ax_original.axvline(x=0, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    ax_original.axvline(x=T, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    st.pyplot(fig_original)

    st.markdown("---")

    # Configuración de armónicos
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Análisis de Fourier")
    N = st.sidebar.slider("Número de armónicos (N):", min_value=1, max_value=50, value=10, step=1)

    c_n, n_values, a0, an_list, bn_list = coef_func(N)

    # Espectro en línea
    st.subheader("📊 Espectro en Línea")
    fig_espectro, ax_espectro = plt.subplots(figsize=(14, 6))
    markerline, stemlines, baseline = ax_espectro.stem(n_values, c_n, basefmt=' ')
    markerline.set_markerfacecolor('blue')
    markerline.set_markeredgecolor('blue')
    markerline.set_markersize(8)
    stemlines.set_color('blue')
    stemlines.set_linewidth(2)
    ax_espectro.set_xlabel('Armónico (n)', fontsize=12)
    ax_espectro.set_ylabel('Amplitud', fontsize=12)
    ax_espectro.set_title('Espectro en Línea', fontsize=14, fontweight='bold')
    ax_espectro.grid(True, alpha=0.3)
    ax_espectro.axhline(y=0, color='k', linewidth=0.8)
    ax_espectro.set_xlim(-0.5, N+1)
    st.pyplot(fig_espectro)

    st.markdown("---")

    # Preparar datos para reconstrucción
    delta = 0.01
    ti = -T
    tf = T
    tiempo = np.arange(ti, tf + delta, delta)

    if tipo_senal == "Función definida en [-1,1] (Ej. 3.6.4)":
        y_reconstruida = reconstruir_senal(an_list, bn_list, tiempo, a0, usar_pi=True)
    else:
        y_reconstruida = reconstruir_senal(an_list, bn_list, tiempo, a0, usar_pi=False)

    y_original_recon = senal_func(tiempo, T)

    # Crear dos columnas para las gráficas
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Señal Original")
        fig_orig, ax_orig = plt.subplots(figsize=(7, 5))
        ax_orig.plot(tiempo, y_original_recon, 'b-', linewidth=2.5, label='Señal Original')
        ax_orig.set_xlabel('Tiempo (t)', fontsize=11)
        ax_orig.set_ylabel('x(t)', fontsize=11)
        ax_orig.set_title('Señal Original', fontsize=12, fontweight='bold')
        ax_orig.legend(fontsize=10, loc='best')
        ax_orig.grid(True, alpha=0.3)
        ax_orig.axhline(y=0, color='k', linewidth=0.5)
        ax_orig.axvline(x=0, color='k', linewidth=0.5)
        ax_orig.axvline(x=-T, color='gray', linewidth=1, linestyle='--', alpha=0.5)
        ax_orig.axvline(x=0, color='gray', linewidth=1, linestyle='--', alpha=0.5)
        ax_orig.axvline(x=T, color='gray', linewidth=1, linestyle='--', alpha=0.5)
        st.pyplot(fig_orig)

    with col2:
        st.subheader("🔄 Señal Reconstruida")
        fig_rec, ax_rec = plt.subplots(figsize=(7, 5))
        ax_rec.plot(tiempo, y_reconstruida, 'r-', linewidth=2.5, label=f'Reconstrucción (N = {N})')
        ax_rec.set_xlabel('Tiempo (t)', fontsize=11)
        ax_rec.set_ylabel('x(t)', fontsize=11)
        ax_rec.set_title(f'Reconstrucción con {N} armónicos', fontsize=12, fontweight='bold')
        ax_rec.legend(fontsize=10, loc='best')
        ax_rec.grid(True, alpha=0.3)
        ax_rec.axhline(y=0, color='k', linewidth=0.5)
        ax_rec.axvline(x=0, color='k', linewidth=0.5)
        ax_rec.axvline(x=-T, color='gray', linewidth=1, linestyle='--', alpha=0.5)
        ax_rec.axvline(x=0, color='gray', linewidth=1, linestyle='--', alpha=0.5)
        ax_rec.axvline(x=T, color='gray', linewidth=1, linestyle='--', alpha=0.5)
        st.pyplot(fig_rec)

    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Tip:** Aumenta N para incluir más armónicos y mejorar la aproximación.")

    with st.expander("📐 Ver fórmulas de los coeficientes"):
        if tipo_senal == "Señal triangular (Ej. 3.6.1)":
            st.latex(r"x(t) = \sum_{n=1}^{\infty} \frac{4}{n^2\pi^2}(1 - \cos(n\pi))\cos(nt)")
            st.write("**Coeficientes:** aₙ = (4/n²π²)(1 - cos(nπ)), bₙ = 0")
        elif tipo_senal == "Señal rampa (Ej. 3.6.2)":
            st.latex(r"x(t) = \sum_{n=1}^{\infty} \frac{-2}{n}\cos(n\pi)\sin(nt)")
            st.write("**Coeficientes:** aₙ = 0, bₙ = (-2/n)cos(nπ)")
        elif tipo_senal == "Función cuadrática (Ej. 3.6.3)":
            st.latex(r"x(t) = \frac{\pi^2}{3} + \sum_{n=1}^{\infty} \frac{4}{n^2}\cos(n\pi)\cos(nt)")
            st.write("**Coeficientes:** a₀ = π²/3, aₙ = (4/n²)cos(nπ), bₙ = 0")
        else:
            st.latex(r"x(t) = \frac{1}{4} + \sum_{n=1}^{\infty} \left[a_n\cos(n\pi t) + b_n\sin(n\pi t)\right]")
            st.write("**Coeficientes:** a₀ = 1/4")
            st.write("**aₙ:** aₙ = 0 para n par, aₙ = 2/(n²π²) para n impar")
            st.write("**bₙ:** bₙ = -1/(nπ) para n par, bₙ = 3/(nπ) para n impar")
# ============================================================================
# PUNTO 2: MODULACIÓN AM
# ============================================================================

elif pagina == "Punto 2: Modulación AM":
    st.header("📡 Punto 2: Modulación y Demodulación con Detección Sincrónica")
    st.markdown("**Implementación según Figuras 1 y 2 del laboratorio**")
    
    # Configuración de parámetros
    st.sidebar.header("⚙️ Parámetros de Configuración")
    
    # Parámetros de la portadora
    st.sidebar.subheader("🌊 Señal Portadora")
    Ac = st.sidebar.slider("Amplitud de portadora (Ac)", 0.5, 2.0, 1.0, 0.1)
    fc = st.sidebar.slider("Frecuencia portadora fc (Hz)", 5000, 20000, 10000, 1000)
    
    # Parámetro del filtro pasa bajas
    st.sidebar.subheader("🔧 Filtro Pasa Bajas")
    cutoff = st.sidebar.slider("Frecuencia de corte del FPB (Hz)", 1000, 8000, 5000, 500)
    
    # Variables para almacenar las señales
    x_t = None
    fs = None
    
    st.sidebar.subheader("📁 Cargar archivo de audio")
    audio_file = st.sidebar.file_uploader("Cargar audio WAV", type=['wav'], key='audio_p2')
    
    if audio_file is not None:
        # Leer archivo de audio
        fs, x_t_raw = wavfile.read(audio_file)
        
        # Combinar canales estéreo a mono
        if x_t_raw.ndim == 2:
            x_t_raw = np.mean(x_t_raw, axis=1)
        
        # Normalizar
        x_t_raw = x_t_raw.astype(float) / np.max(np.abs(x_t_raw))
        
        # Duración en el tiempo
        n = len(x_t_raw)
        dur_aud = n / fs
        ts = 1 / fs
        t = np.arange(n) * ts
        
        st.success(f"✅ Archivo de audio cargado correctamente")
        st.info(f"**Frecuencia de muestreo:** {fs} Hz | **Duración:** {dur_aud:.2f} s | **Muestras:** {n}")
        
        # ========== ANÁLISIS DE LA SEÑAL ORIGINAL ==========
        st.header("📊 1. Análisis de la Señal de Audio Original x(t)")
        
        # Calcular FFT
        x_f = np.fft.fft(x_t_raw)
        x_fcent = np.fft.fftshift(x_f)
        delta_f = 1 / (n * ts)
        f = np.arange(-n/2, n/2) * delta_f
        
        # Magnitud del espectro
        dep_original = np.abs(x_fcent / n)
        
        # Reproducir audio original
        def audio_player(audio_data, sample_rate, label):
            audio_normalized = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
            buffer = io.BytesIO()
            wavfile.write(buffer, sample_rate, audio_normalized)
            audio_base64 = base64.b64encode(buffer.getvalue()).decode()
            audio_html = f"""
            <div style="margin: 10px 0;">
                <p><strong>{label}</strong></p>
                <audio controls style="width: 100%;">
                    <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
                </audio>
            </div>
            """
            return audio_html
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 4))
            t_display = min(0.05, dur_aud)
            idx_display = int(t_display * fs)
            ax.plot(t[:idx_display], x_t_raw[:idx_display], 'b', linewidth=1.5)
            ax.set_xlabel('Tiempo (s)')
            ax.set_ylabel('Amplitud')
            ax.set_title('Señal de Audio x(t) en el Tiempo')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(f, dep_original, 'b', linewidth=1.5)
            ax.set_xlabel('Frecuencia (Hz)')
            ax.set_ylabel('Magnitud')
            ax.set_title('Espectro de x(t) - Magnitud')
            ax.set_xlim([-fs/2, fs/2])
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        st.markdown(audio_player(x_t_raw, fs, "🎵 Audio Original x(t)"), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ========== FILTRADO DE LA SEÑAL ==========
        st.header("🔧 2. Filtrado Pasa Bajas - Limitación de Ancho de Banda")
        
        st.info(f"📏 **Frecuencia de corte seleccionada:** {cutoff} Hz")
        
        # ✅ CORRECCIÓN: Crear filtro pasa bajas ideal correctamente
        fpb = (np.abs(f) <= cutoff).astype(float)
        
        # Aplicar filtro en dominio de frecuencia
        x_f_fil = x_fcent * fpb
        dep_filtrada = np.abs(x_f_fil / n)
        
        # ✅ CORRECCIÓN: Regresar al dominio del tiempo correctamente
        x_f_filco = np.fft.ifftshift(x_f_fil)
        x_t = np.real(np.fft.ifft(x_f_filco))
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(f, fpb, 'r', linewidth=2)
            ax.set_xlabel('Frecuencia (Hz)')
            ax.set_ylabel('H(f)')
            ax.set_title(f'Filtro Pasa Bajas Ideal (fc = {cutoff} Hz)')
            ax.set_xlim([-fs/2, fs/2])
            ax.set_ylim([-0.1, 1.1])
            ax.axvline(cutoff, color='g', linestyle='--', alpha=0.5, label=f'+{cutoff} Hz')
            ax.axvline(-cutoff, color='g', linestyle='--', alpha=0.5, label=f'-{cutoff} Hz')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(f, dep_original, 'b', alpha=0.5, linewidth=1.5, label='Original')
            ax.plot(f, dep_filtrada, 'r', linewidth=2, label='Filtrada')
            ax.set_xlabel('Frecuencia (Hz)')
            ax.set_ylabel('Magnitud')
            ax.set_title('Comparación de Espectros')
            ax.set_xlim([-10000, 10000])
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Comparación temporal: Original vs Filtrada
        st.subheader("📊 Comparación Temporal: Original vs Filtrada")
        fig, ax = plt.subplots(figsize=(12, 4))
        idx_comp = slice(0, min(int(1*fs), n))
        ax.plot(t[idx_comp], x_t_raw[idx_comp], label='Original', linewidth=2, alpha=0.6, color='#3b82f6')
        ax.plot(t[idx_comp], x_t[idx_comp], label=f'Filtrada ({cutoff}Hz)', linewidth=2, color='#ef4444')
        ax.set_xlabel("Tiempo (s)", fontsize=11, fontweight='bold')
        ax.set_ylabel("Amplitud", fontsize=11, fontweight='bold')
        ax.set_title("Señal Original vs Señal Filtrada", fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        col3, col4 = st.columns(2)
        
        with col3:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(t[:idx_display], x_t[:idx_display], 'r', linewidth=1.5)
            ax.set_xlabel('Tiempo (s)')
            ax.set_ylabel('Amplitud')
            ax.set_title('Señal Filtrada x(t) - Zoom')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col4:
            st.markdown(audio_player(x_t, fs, "🎵 Audio Filtrado x(t)"), unsafe_allow_html=True)
            st.markdown("""
            **Nota:** Compare el audio original con el filtrado. 
            Si la pérdida de información es notoria, ajuste la frecuencia de corte.
            """)
        
        st.markdown("---")
        
        # ========== PROCESO DE MODULACIÓN ==========
        st.header("📡 3. Proceso de Modulación (Figura 1)")
        
        # Generar portadora
        carrier_cos = Ac * np.cos(2 * np.pi * fc * t)
        
        # Calcular espectro de la portadora
        carrier_f = np.fft.fft(carrier_cos)
        carrier_fcent = np.fft.fftshift(carrier_f)
        dep_carrier = np.abs(carrier_fcent / n)
        
        # Mostrar portadora
        st.subheader("🌊 Señal Portadora: cos(2πfct)")
        
        st.latex(r"c(t) = A_c \cos(2\pi f_c t)")
        st.info(f"Portadora: Amplitud Ac = {Ac}, Frecuencia fc = {fc} Hz")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 4))
            # Mostrar más ciclos de la portadora para mejor visualización
            t_carrier_display = min(20/fc, dur_aud)  # Mostrar ~20 ciclos
            idx_carrier_display = int(t_carrier_display * fs)
            ax.plot(t[:idx_carrier_display], carrier_cos[:idx_carrier_display], 'orange', linewidth=1.5)
            ax.set_xlabel('Tiempo (s)')
            ax.set_ylabel('Amplitud')
            ax.set_title(f'Portadora c(t) = {Ac}·cos(2π·{fc}·t)')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(f, dep_carrier, 'orange', linewidth=2)
            ax.set_xlabel('Frecuencia (Hz)')
            ax.set_ylabel('Magnitud')
            ax.set_title('Espectro de la Portadora')
            ax.set_xlim([-(fc+5000), (fc+5000)])
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")
        
        # 🔴 Punto Rojo 1: Señal moduladora x(t)
        st.subheader("🔴 Punto 1: x(t) - Señal moduladora (ya filtrada)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(t[:idx_display], x_t[:idx_display], 'b', linewidth=1.5)
            ax.set_xlabel('Tiempo (s)')
            ax.set_ylabel('Amplitud')
            ax.set_title('x(t) en el tiempo')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(f, dep_filtrada, 'b', linewidth=1.5)
            ax.set_xlabel('Frecuencia (Hz)')
            ax.set_ylabel('Magnitud')
            ax.set_title('X(ω) - Espectro en banda base')
            ax.set_xlim([-cutoff*3, cutoff*3])
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")
        
        # 🔴 Punto Rojo 2: Señal modulada y(t)
        y_t = x_t * carrier_cos
        
        # Calcular espectro de y(t)
        y_f = np.fft.fft(y_t)
        y_fcent = np.fft.fftshift(y_f)
        dep_y = np.abs(y_fcent / n)
        
        st.subheader("🔴 Punto 2: y(t) = x(t)cos(ωₒt) - Señal modulada")
        
        st.latex(r"y(t) = x(t)\cos(\omega_o t)")
        st.latex(r"Y(\omega) = \frac{1}{2}[X(\omega - \omega_o) + X(\omega + \omega_o)]")
        
        st.info("💡 **Observe cómo el espectro se traslada:** La señal de banda base (centrada en 0 Hz) ahora aparece centrada en ±fc")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(t[:idx_display], y_t[:idx_display], 'g', linewidth=1.5)
            ax.set_xlabel('Tiempo (s)')
            ax.set_ylabel('Amplitud')
            ax.set_title('y(t) = x(t)cos(ωₒt) en el tiempo')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(f, dep_y, 'g', linewidth=1.5)
            ax.set_xlabel('Frecuencia (Hz)')
            ax.set_ylabel('Magnitud')
            ax.set_title('Y(ω) - Espectro de la señal modulada')
            ax.set_xlim([-(fc + cutoff*3), (fc + cutoff*3)])
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Gráfica adicional con zoom en banda lateral superior
        st.subheader("🔍 Vista ampliada: Bandas Laterales")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
        
        # Banda lateral superior (alrededor de +fc)
        ax1.plot(f, dep_y, 'g', linewidth=1.5)
        ax1.set_xlabel('Frecuencia (Hz)')
        ax1.set_ylabel('Magnitud')
        ax1.set_title(f'Banda Lateral Superior')
        ax1.set_xlim([fc - cutoff*1.5, fc + cutoff*1.5])
        ax1.grid(True, alpha=0.3)
        
        # Banda lateral inferior (alrededor de -fc)
        ax2.plot(f, dep_y, 'g', linewidth=1.5)
        ax2.set_xlabel('Frecuencia (Hz)')
        ax2.set_ylabel('Magnitud')
        ax2.set_title(f'Banda Lateral Inferior')
        ax2.set_xlim([-fc - cutoff*1.5, -fc + cutoff*1.5])
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
        
        # ========== PROCESO DE DEMODULACIÓN ==========
        st.header("📥 4. Proceso de Demodulación (Figura 2)")
        
        # 🔴 Punto Rojo 3: Después de multiplicar por cos(ωₒt)
        x_prime_t = y_t * carrier_cos
        
        # Calcular espectro de x'(t)
        xp_f = np.fft.fft(x_prime_t)
        xp_fcent = np.fft.fftshift(xp_f)
        dep_xp = np.abs(xp_fcent / n)
        
        st.subheader("🔴 Punto 3: x'(t) = y(t)cos(ωₒt) - Después del multiplicador")
        
        st.latex(r"x'(t) = y(t)\cos(\omega_o t) = x(t)\cos^2(\omega_o t)")
        st.latex(r"X'(\omega) = \frac{1}{2}X(\omega) + \frac{1}{4}[X(\omega - 2\omega_o) + X(\omega + 2\omega_o)]")
        
        st.info("💡 **Observe:** Ahora hay componentes en banda base (0 Hz) y en ±2fc")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(t[:idx_display], x_prime_t[:idx_display], 'purple', linewidth=1.5)
            ax.set_xlabel('Tiempo (s)')
            ax.set_ylabel('Amplitud')
            ax.set_title("x'(t) antes del filtro")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(f, dep_xp, 'purple', linewidth=1.5)
            ax.set_xlabel('Frecuencia (Hz)')
            ax.set_ylabel("Magnitud")
            ax.set_title("X'(ω) - Espectro después de la multiplicación")
            ax.set_xlim([-(2*fc + cutoff*2), (2*fc + cutoff*2)])
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")
        
        # 🔴 Punto Rojo 4: Después del FPB
        # Aplicar filtro pasa bajas ideal en frecuencia
        xp_f_fil = xp_fcent * fpb
        
        # Regresar al tiempo
        xp_f_filco = np.fft.ifftshift(xp_f_fil)
        x_recovered = np.real(np.fft.ifft(xp_f_filco)) * 2  # Multiplicar por 2
        
        # Calcular espectro recuperado
        dep_rec = np.abs(xp_f_fil / n)
        
        st.subheader("🔴 Punto 4: x(t) recuperada - Después del FPB")
        
        st.info("💡 **El filtro elimina las componentes en ±2fc y deja solo la señal en banda base**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(t[:idx_display], x_recovered[:idx_display], 'b', linewidth=2, label='Recuperada')
            ax.plot(t[:idx_display], x_t[:idx_display], 'r--', alpha=0.5, linewidth=1.5, label='Original')
            ax.set_xlabel('Tiempo (s)')
            ax.set_ylabel('Amplitud')
            ax.set_title('Señal Recuperada vs Original')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(f, dep_rec, 'b', linewidth=2, label='Recuperada')
            ax.plot(f, dep_filtrada, 'r--', alpha=0.5, linewidth=1.5, label='Original')
            ax.set_xlabel('Frecuencia (Hz)')
            ax.set_ylabel('Magnitud')
            ax.set_title('Espectro: Señal recuperada')
            ax.set_xlim([-cutoff*2, cutoff*2])
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")
        
        # ========== COMPARACIÓN FINAL ==========
        st.header("🎧 5. Comparación Final y Reproducción de Audio")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(audio_player(x_t, fs, "🎵 Señal Original x(t) (filtrada)"), unsafe_allow_html=True)
        
        with col2:
            st.markdown(audio_player(x_recovered, fs, "🎵 Señal Recuperada"), unsafe_allow_html=True)
        
        st.markdown(audio_player(y_t, fs, "📡 Señal Modulada y(t)"), unsafe_allow_html=True)
        
        # Resumen teórico
        with st.expander("📚 Resumen Teórico del Proceso"):
            st.markdown("""
            ### Proceso de Modulación AM:
            
            **1. Señal Original:** X(ω) centrada en 0 Hz (banda base)
            
            **2. Modulación:** y(t) = x(t)cos(ωₒt)
            - El espectro se **traslada** a ±fc
            - Y(ω) = ½[X(ω - ωₒ) + X(ω + ωₒ)]
            - Aparecen dos bandas laterales simétricas alrededor de ±fc
            
            **3. Demodulación:** x'(t) = y(t)cos(ωₒt)
            - Multiplicamos nuevamente por la portadora
            - X'(ω) = ½X(ω) + ¼[X(ω - 2ωₒ) + X(ω + 2ωₒ)]
            - Componente deseada en banda base (0 Hz)
            - Componentes no deseadas en ±2fc
            
            **4. Filtrado:** El FPB elimina las componentes en ±2fc
            - Solo queda ½X(ω) en banda base
            - Se multiplica por 2 para recuperar la amplitud original
            
            ### Condiciones importantes:
            - **fc >> cutoff**: La frecuencia portadora debe ser mucho mayor que el ancho de banda de la señal
            - **Frecuencia de Nyquist**: fs/2 > 2fc + cutoff para evitar aliasing
            """)
        
        st.markdown("""
        ### 📝 Instrucciones de Uso:
        
        1. **Cargar Audio:** Sube un archivo WAV en el panel lateral
        2. **Análisis Inicial:** Observa la señal original en tiempo y frecuencia
        3. **Ajustar Filtro:** Selecciona la frecuencia de corte del FPB para limitar el ancho de banda
        4. **Configurar Portadora:** Ajusta la amplitud y frecuencia de la señal portadora (fc debe ser >> cutoff)
        5. **Observar Traslación:** Verifica cómo el espectro se traslada a ±fc en la modulación
        6. **Escuchar Resultados:** Compara el audio original con el recuperado
        """)
    
    else:
        st.info("👈 Por favor, carga un archivo de audio WAV desde el panel lateral para comenzar.")

# ============================================================================
# PUNTO 3: MODULACIÓN Y DEMODULACIÓN EN CUADRATURA DE FASE
# ============================================================================

elif pagina == "Punto 3: Modulación y Demodulación en cuadratura de fase":
    st.header("📡 Punto 3: Multiplexación en Cuadratura (Figura 3)")
    st.markdown("**Transmisión simultánea de dos señales senoidales usando ortogonalidad de senos y cosenos**")
    
    # Configuración de parámetros
    st.sidebar.header("⚙️ Parámetros de Configuración")
    
    # Parámetros de las señales moduladoras
    st.sidebar.subheader("🌊 Señales Moduladoras")
    f1 = st.sidebar.slider("Frecuencia de x₁(t) (Hz)", 1, 50, 5, 1)
    A1 = st.sidebar.slider("Amplitud de x₁(t)", 0.1, 2.0, 1.0, 0.1)
    
    f2 = st.sidebar.slider("Frecuencia de x₂(t) (Hz)", 1, 50, 10, 1)
    A2 = st.sidebar.slider("Amplitud de x₂(t)", 0.1, 2.0, 1.0, 0.1)
    
    # Parámetros de la portadora
    st.sidebar.subheader("📻 Portadora")
    fc = st.sidebar.slider("Frecuencia portadora fc (Hz)", 100, 500, 200, 10)
    
    # Parámetro del filtro
    st.sidebar.subheader("🔧 Filtro Pasa Bajas")
    fc_filter = st.sidebar.slider("Frecuencia de corte FPB (Hz)", 10, 100, 50, 5)
    
    # Parámetros de simulación
    st.sidebar.subheader("⏱️ Tiempo de Simulación")
    T_total = st.sidebar.slider("Duración (segundos)", 0.5, 5.0, 2.0, 0.5)
    fs = st.sidebar.slider("Frecuencia de muestreo (Hz)", 1000, 5000, 2000, 500)
    
    # Generar vector de tiempo
    t = np.linspace(0, T_total, int(fs * T_total))
    n = len(t)
    ts = 1 / fs
    
    # Generar señales moduladoras senoidales
    x1_t = A1 * np.sin(2 * np.pi * f1 * t)
    x2_t = A2 * np.sin(2 * np.pi * f2 * t)
    
    # Generar portadoras en cuadratura
    cos_carrier = np.cos(2 * np.pi * fc * t)
    sin_carrier = np.sin(2 * np.pi * fc * t)
    
    # Configuración del eje de frecuencia
    delta_f = 1 / (n * ts)
    f = np.arange(-n/2, n/2) * delta_f
    
    st.success(f"✅ Simulación configurada: {n} muestras | fs = {fs} Hz | Duración = {T_total} s")
    
    # ========== SEÑALES ORIGINALES ==========
    st.header("📊 Señales Moduladoras Originales")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Señal x₁(t)")
        
        # Tiempo
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=t, y=x1_t, mode='lines', name='x₁(t)', 
                                  line=dict(color='blue', width=2)))
        fig1.update_layout(
            title=f'x₁(t) = {A1}·sin(2π·{f1}·t)',
            xaxis_title='Tiempo (s)',
            yaxis_title='Amplitud',
            height=350,
            hovermode='x unified'
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Frecuencia
        x1_f = np.fft.fft(x1_t)
        x1_fcent = np.fft.fftshift(x1_f)
        mag_x1 = np.abs(x1_fcent/n)
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=f, y=mag_x1, mode='lines', name='|X₁(f)|',
                                  line=dict(color='blue', width=2)))
        fig2.update_layout(
            title='Espectro de x₁(t)',
            xaxis_title='Frecuencia (Hz)',
            yaxis_title='Magnitud normalizada',
            height=350,
            xaxis_range=[-100, 100],
            hovermode='x unified'
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.subheader("Señal x₂(t)")
        
        # Tiempo
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=t, y=x2_t, mode='lines', name='x₂(t)',
                                  line=dict(color='red', width=2)))
        fig3.update_layout(
            title=f'x₂(t) = {A2}·sin(2π·{f2}·t)',
            xaxis_title='Tiempo (s)',
            yaxis_title='Amplitud',
            height=350,
            hovermode='x unified'
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        # Frecuencia
        x2_f = np.fft.fft(x2_t)
        x2_fcent = np.fft.fftshift(x2_f)
        mag_x2 = np.abs(x2_fcent/n)
        
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=f, y=mag_x2, mode='lines', name='|X₂(f)|',
                                  line=dict(color='red', width=2)))
        fig4.update_layout(
            title='Espectro de x₂(t)',
            xaxis_title='Frecuencia (Hz)',
            yaxis_title='Magnitud normalizada',
            height=350,
            xaxis_range=[-100, 100],
            hovermode='x unified'
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    st.markdown("---")
    
    # ========== SEÑALES PORTADORAS ==========
    st.header("📻 Señales Portadoras en Cuadratura")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Portadora cos(ωc·t)")
        
        # Tiempo
        fig_cos = go.Figure()
        fig_cos.add_trace(go.Scatter(x=t, y=cos_carrier, mode='lines', name='cos(ωc·t)',
                                     line=dict(color='green', width=2)))
        fig_cos.update_layout(
            title=f'cos(2π·{fc}·t)',
            xaxis_title='Tiempo (s)',
            yaxis_title='Amplitud',
            height=350,
            hovermode='x unified'
        )
        st.plotly_chart(fig_cos, use_container_width=True)
        
        # Frecuencia
        cos_f = np.fft.fft(cos_carrier)
        cos_fcent = np.fft.fftshift(cos_f)
        mag_cos = np.abs(cos_fcent/n)
        
        fig_cos_f = go.Figure()
        fig_cos_f.add_trace(go.Scatter(x=f, y=mag_cos, mode='lines', name='|Cos(f)|',
                                       line=dict(color='green', width=2)))
        fig_cos_f.update_layout(
            title='Espectro de cos(ωc·t)',
            xaxis_title='Frecuencia (Hz)',
            yaxis_title='Magnitud normalizada',
            height=350,
            xaxis_range=[-(fc+100), (fc+100)],
            hovermode='x unified'
        )
        st.plotly_chart(fig_cos_f, use_container_width=True)
    
    with col2:
        st.subheader("Portadora sin(ωc·t)")
        
        # Tiempo
        fig_sin = go.Figure()
        fig_sin.add_trace(go.Scatter(x=t, y=sin_carrier, mode='lines', name='sin(ωc·t)',
                                     line=dict(color='orange', width=2)))
        fig_sin.update_layout(
            title=f'sin(2π·{fc}·t)',
            xaxis_title='Tiempo (s)',
            yaxis_title='Amplitud',
            height=350,
            hovermode='x unified'
        )
        st.plotly_chart(fig_sin, use_container_width=True)
        
        # Frecuencia
        sin_f = np.fft.fft(sin_carrier)
        sin_fcent = np.fft.fftshift(sin_f)
        mag_sin = np.abs(sin_fcent/n)
        
        fig_sin_f = go.Figure()
        fig_sin_f.add_trace(go.Scatter(x=f, y=mag_sin, mode='lines', name='|Sin(f)|',
                                       line=dict(color='orange', width=2)))
        fig_sin_f.update_layout(
            title='Espectro de sin(ωc·t)',
            xaxis_title='Frecuencia (Hz)',
            yaxis_title='Magnitud normalizada',
            height=350,
            xaxis_range=[-(fc+100), (fc+100)],
            hovermode='x unified'
        )
        st.plotly_chart(fig_sin_f, use_container_width=True)
    
    st.markdown("---")
    
    # ========== MODULACIÓN ==========
    st.header("📡 MODULACIÓN - Lado Izquierdo de la Figura 3")
    
    # 🔴 Punto Rojo 1: x₁(t) * cos(ωc·t)
    y1_t = x1_t * cos_carrier
    
    st.subheader("🔴 Punto Rojo 1: y₁(t) = x₁(t)·cos(ωc·t)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=y1_t, mode='lines', name='y₁(t)',
                                line=dict(color='blue', width=1.5)))
        fig.update_layout(
            title='y₁(t) en el tiempo',
            xaxis_title='Tiempo (s)',
            yaxis_title='Amplitud',
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        y1_f = np.fft.fft(y1_t)
        y1_fcent = np.fft.fftshift(y1_f)
        mag_y1 = np.abs(y1_fcent/n)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=f, y=mag_y1, mode='lines', name='|Y₁(f)|',
                                line=dict(color='blue', width=1.5)))
        fig.update_layout(
            title='Espectro de y₁(t)',
            xaxis_title='Frecuencia (Hz)',
            yaxis_title='Magnitud normalizada',
            height=400,
            xaxis_range=[-(fc+100), (fc+100)],
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 🔴 Punto Rojo 2: x₂(t) * sin(ωc·t)
    y2_t = x2_t * sin_carrier
    
    st.subheader("🔴 Punto Rojo 2: y₂(t) = x₂(t)·sin(ωc·t)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=y2_t, mode='lines', name='y₂(t)',
                                line=dict(color='red', width=1.5)))
        fig.update_layout(
            title='y₂(t) en el tiempo',
            xaxis_title='Tiempo (s)',
            yaxis_title='Amplitud',
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        y2_f = np.fft.fft(y2_t)
        y2_fcent = np.fft.fftshift(y2_f)
        mag_y2 = np.abs(y2_fcent/n)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=f, y=mag_y2, mode='lines', name='|Y₂(f)|',
                                line=dict(color='red', width=1.5)))
        fig.update_layout(
            title='Espectro de y₂(t)',
            xaxis_title='Frecuencia (Hz)',
            yaxis_title='Magnitud normalizada',
            height=400,
            xaxis_range=[-(fc+100), (fc+100)],
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 🔴 Punto Rojo 3: Suma Σ
    y_sum_t = y1_t + y2_t
    
    st.subheader("🔴 Punto Rojo 3: Señal Transmitida = y₁(t) + y₂(t)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=y_sum_t, mode='lines', name='y(t) transmitida',
                                line=dict(color='purple', width=1.5)))
        fig.update_layout(
            title='Señal transmitida en el tiempo',
            xaxis_title='Tiempo (s)',
            yaxis_title='Amplitud',
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        ysum_f = np.fft.fft(y_sum_t)
        ysum_fcent = np.fft.fftshift(ysum_f)
        mag_ysum = np.abs(ysum_fcent/n)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=f, y=mag_ysum, mode='lines', name='|Y(f)|',
                                line=dict(color='purple', width=1.5)))
        fig.update_layout(
            title='Espectro de la señal transmitida',
            xaxis_title='Frecuencia (Hz)',
            yaxis_title='Magnitud normalizada',
            height=400,
            xaxis_range=[-(fc+100), (fc+100)],
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ========== DEMODULACIÓN ==========
    st.header("📥 DEMODULACIÓN - Lado Derecho de la Figura 3")
    
    st.info("""
    **Proceso de recuperación usando ortogonalidad:**
    - Para recuperar x₁(t): multiplicar por cos(ωc·t) y aplicar FPB
    - Para recuperar x₂(t): multiplicar por sin(ωc·t) y aplicar FPB
    """)
    
    # Filtro pasa-bajas ideal
    fpb = np.abs(f) <= fc_filter
    
    # ========== RECUPERACIÓN DE x₁(t) ==========
    st.subheader("🔵 Recuperación de x₁(t)")
    
    # 🔴 Multiplicar señal recibida por cos(ωc·t)
    x1_prime_t = y_sum_t * cos_carrier
    
    st.markdown("🔴 Después de multiplicar por cos(ωc·t)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=x1_prime_t, mode='lines', name="x₁'(t)",
                                line=dict(color='cyan', width=1.5)))
        fig.update_layout(
            title="x₁'(t) antes del filtro",
            xaxis_title='Tiempo (s)',
            yaxis_title='Amplitud',
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        x1p_f = np.fft.fft(x1_prime_t)
        x1p_fcent = np.fft.fftshift(x1p_f)
        mag_x1p = np.abs(x1p_fcent/n)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=f, y=mag_x1p, mode='lines', name="|X₁'(f)|",
                                line=dict(color='cyan', width=1.5)))
        fig.update_layout(
            title="Espectro de x₁'(t)",
            xaxis_title='Frecuencia (Hz)',
            yaxis_title='Magnitud normalizada',
            height=400,
            xaxis_range=[-(2*fc+100), (2*fc+100)],
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 🔴 Aplicar FPB y recuperar
    x1p_f_fil = x1p_fcent * fpb
    x1p_f_filco = np.fft.ifftshift(x1p_f_fil)
    x1_rec_t = np.real(np.fft.ifft(x1p_f_filco)) * 2
    
    st.markdown("✅ Después del FPB - Señal x₁(t) recuperada")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=x1_t, mode='lines', name='Original',
                                line=dict(color='blue', width=2, dash='dash'), opacity=0.6))
        fig.add_trace(go.Scatter(x=t, y=x1_rec_t, mode='lines', name='Recuperada',
                                line=dict(color='green', width=2)))
        fig.update_layout(
            title='x₁(t) recuperada vs original',
            xaxis_title='Tiempo (s)',
            yaxis_title='Amplitud',
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        x1rec_f = np.fft.fft(x1_rec_t)
        x1rec_fcent = np.fft.fftshift(x1rec_f)
        mag_x1rec = np.abs(x1rec_fcent/n)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=f, y=mag_x1, mode='lines', name='Original',
                                line=dict(color='blue', width=2, dash='dash'), opacity=0.6))
        fig.add_trace(go.Scatter(x=f, y=mag_x1rec, mode='lines', name='Recuperada',
                                line=dict(color='green', width=2)))
        fig.update_layout(
            title='Espectro comparativo - x₁(t)',
            xaxis_title='Frecuencia (Hz)',
            yaxis_title='Magnitud normalizada',
            height=400,
            xaxis_range=[-100, 100],
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

    
    st.markdown("---")
    
    # ========== RECUPERACIÓN DE x₂(t) ==========
    st.subheader("🔴 Recuperación de x₂(t)")
    
    # 🔴 Multiplicar señal recibida por sin(ωc·t)
    x2_prime_t = y_sum_t * sin_carrier
    
    st.markdown("🔴 Después de multiplicar por sin(ωc·t)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=x2_prime_t, mode='lines', name="x₂'(t)",
                                line=dict(color='orange', width=1.5)))
        fig.update_layout(
            title="x₂'(t) antes del filtro",
            xaxis_title='Tiempo (s)',
            yaxis_title='Amplitud',
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        x2p_f = np.fft.fft(x2_prime_t)
        x2p_fcent = np.fft.fftshift(x2p_f)
        mag_x2p = np.abs(x2p_fcent/n)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=f, y=mag_x2p, mode='lines', name="|X₂'(f)|",
                                line=dict(color='orange', width=1.5)))
        fig.update_layout(
            title="Espectro de x₂'(t)",
            xaxis_title='Frecuencia (Hz)',
            yaxis_title='Magnitud normalizada',
            height=400,
            xaxis_range=[-(2*fc+100), (2*fc+100)],
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 🔴 Aplicar FPB y recuperar
    x2p_f_fil = x2p_fcent * fpb
    x2p_f_filco = np.fft.ifftshift(x2p_f_fil)
    x2_rec_t = np.real(np.fft.ifft(x2p_f_filco)) * 2
    
    st.markdown("✅ Después del FPB - Señal x₂(t) recuperada")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=x2_t, mode='lines', name='Original',
                                line=dict(color='red', width=2, dash='dash'), opacity=0.6))
        fig.add_trace(go.Scatter(x=t, y=x2_rec_t, mode='lines', name='Recuperada',
                                line=dict(color='green', width=2)))
        fig.update_layout(
            title='x₂(t) recuperada vs original',
            xaxis_title='Tiempo (s)',
            yaxis_title='Amplitud',
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        x2rec_f = np.fft.fft(x2_rec_t)
        x2rec_fcent = np.fft.fftshift(x2rec_f)
        mag_x2rec = np.abs(x2rec_fcent/n)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=f, y=mag_x2, mode='lines', name='Original',
                                line=dict(color='red', width=2, dash='dash'), opacity=0.6))
        fig.add_trace(go.Scatter(x=f, y=mag_x2rec, mode='lines', name='Recuperada',
                                line=dict(color='green', width=2)))
        fig.update_layout(
            title='Espectro comparativo - x₂(t)',
            xaxis_title='Frecuencia (Hz)',
            yaxis_title='Magnitud normalizada',
            height=400,
            xaxis_range=[-100, 100],
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ========== DEMOSTRACIÓN MATEMÁTICA ==========
    st.header("📚 Demostración Matemática")
    
    with st.expander("📖 Ver demostración completa"):
        st.markdown(r"""
        ### Ortogonalidad de Senos y Cosenos
        
        Las señales cos(ωc·t) y sin(ωc·t) son **ortogonales**, es decir:
        
        $$\int_{0}^{T} \cos(\omega_c t) \sin(\omega_c t) \, dt = 0$$
        
        ### Proceso de Modulación
        
        Señal transmitida:
        $$y(t) = x_1(t)\cos(\omega_c t) + x_2(t)\sin(\omega_c t)$$
        
        ### Demodulación de x₁(t)
        
        Multiplicamos por cos(ωc·t):
        $$y(t) \cdot \cos(\omega_c t) = x_1(t)\cos^2(\omega_c t) + x_2(t)\sin(\omega_c t)\cos(\omega_c t)$$
        
        Usando identidades trigonométricas:
        - $\cos^2(\omega_c t) = \frac{1}{2}[1 + \cos(2\omega_c t)]$
        - $\sin(\omega_c t)\cos(\omega_c t) = \frac{1}{2}\sin(2\omega_c t)$
        
        Obtenemos:
        $$y(t) \cdot \cos(\omega_c t) = \frac{x_1(t)}{2}[1 + \cos(2\omega_c t)] + \frac{x_2(t)}{2}\sin(2\omega_c t)$$
        
        Al aplicar el **filtro pasa-bajas** (FPB), eliminamos los términos de alta frecuencia (2ωc):
        $$\text{FPB}\{y(t) \cdot \cos(\omega_c t)\} = \frac{x_1(t)}{2}$$
        
        Multiplicando por 2:
        $$\boxed{x_1(t) \text{ recuperada}}$$
        
        ### Demodulación de x₂(t)
        
        De manera similar, multiplicando por sin(ωc·t):
        $$y(t) \cdot \sin(\omega_c t) = x_1(t)\cos(\omega_c t)\sin(\omega_c t) + x_2(t)\sin^2(\omega_c t)$$
        
        Usando identidades:
        - $\sin^2(\omega_c t) = \frac{1}{2}[1 - \cos(2\omega_c t)]$
        
        Aplicando FPB y multiplicando por 2:
        $$\boxed{x_2(t) \text{ recuperada}}$$
        
        ### Conclusión
        
        Gracias a la **ortogonalidad** de senos y cosenos, las dos señales pueden transmitirse simultáneamente en la misma frecuencia portadora y recuperarse independientemente.
        """)
    
    st.success("""
    ✅ El sistema de multiplexación en cuadratura permite transmitir dos señales senoidales diferentes 
    simultáneamente en la misma frecuencia portadora, aprovechando la ortogonalidad entre cos(ωc·t) y sin(ωc·t).
    """)
    
    # Resumen de parámetros
    with st.expander("📊 Resumen de Parámetros de Simulación"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Señales Moduladoras:**")
            st.write(f"• x₁(t): {A1}·sin(2π·{f1}·t)")
            st.write(f"• x₂(t): {A2}·sin(2π·{f2}·t)")
        
        with col2:
            st.markdown("**Portadora:**")
            st.write(f"• Frecuencia: {fc} Hz")
            st.write(f"• Filtro FPB: {fc_filter} Hz")
        
        with col3:
            st.markdown("**Simulación:**")
            st.write(f"• Duración: {T_total} s")
            st.write(f"• Frecuencia muestreo: {fs} Hz")
            st.write(f"• Número de muestras: {n}")

# ================== PUNTO 4 ==================

elif pagina == "Punto 4: Modulación de amplitud DSB-LC":
    st.title("📡 Modulación de Señales Sinusoidales")
    st.write("Configura los parámetros de las señales y visualiza la modulación AM")

    # Sidebar para inputs
    st.sidebar.header("⚙️ Parámetros de las Señales")

    st.sidebar.subheader("🌊 Señal 1")
    f1 = st.sidebar.number_input("Frecuencia f1 (Hz)", min_value=1.0, max_value=5000.0, value=250.0, step=10.0)
    Amp1 = st.sidebar.number_input("Amplitud A1", min_value=0.1, max_value=5.0, value=1.2, step=0.1)

    st.sidebar.subheader("🌊 Señal 2")
    f2 = st.sidebar.number_input("Frecuencia f2 (Hz)", min_value=1.0, max_value=5000.0, value=500.0, step=10.0)
    Amp2 = st.sidebar.number_input("Amplitud A2", min_value=0.1, max_value=5.0, value=0.8, step=0.1)

    st.sidebar.subheader("🌊 Señal 3")
    f3 = st.sidebar.number_input("Frecuencia f3 (Hz)", min_value=1.0, max_value=5000.0, value=1000.0, step=10.0)
    Amp3 = st.sidebar.number_input("Amplitud A3", min_value=0.1, max_value=5.0, value=0.4, step=0.1)

    st.sidebar.subheader("📻 Portadora")
    f_port = st.sidebar.number_input("Frecuencia Portadora (Hz)", min_value=1000.0, max_value=50000.0, value=12000.0, step=100.0)

    # Parámetros fijos
    fs = 100000
    T = 0.020
    t = np.arange(0, T, 1/fs)
    N = len(t)

    # Frecuencias angulares
    w1 = 2 * np.pi * f1
    w2 = 2 * np.pi * f2
    w3 = 2 * np.pi * f3
    w_port = 2 * np.pi * f_port

    # Generación de señales
    y1 = Amp1 * np.cos(w1 * t)
    y2 = Amp2 * np.cos(w2 * t)
    y3 = Amp3 * np.cos(w3 * t)
    y_t = y1 + y2 + y3
    portadora = np.cos(w_port * t)

    # Configuración FFT centrada (como en el código de referencia)
    fft_freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))

    st.success(f"✅ Configuración: fs = {fs} Hz | T = {T*1000} ms | N = {N} muestras")

    # ========== SEÑALES INDIVIDUALES ==========
    st.header("📊 Señales Senoidales Individuales")
    
    # Señal 1
    st.subheader("🔵 Señal y₁(t)")
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=t*1000, y=y1, mode='lines', name='y₁(t)',
                                  line=dict(color='blue', width=2)))
        fig1.update_layout(
            title=f'y₁(t) = {Amp1}·cos(2π·{f1}·t)',
            xaxis_title='Tiempo (ms)',
            yaxis_title='Amplitud',
            height=400,
            xaxis_range=[0, 6],
            hovermode='x unified'
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        Y1_fft = np.fft.fftshift(np.fft.fft(y1))
        Y1_mag = np.abs(Y1_fft) / N
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=fft_freqs, y=Y1_mag, mode='lines',
                                  line=dict(color='blue', width=2)))
        fig2.update_layout(
            title='Espectro de y₁(t)',
            xaxis_title='Frecuencia (Hz)',
            yaxis_title='Amplitud',
            height=400,
            xaxis_range=[-2000, 2000],
            hovermode='x unified'
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Señal 2
    st.subheader("🟢 Señal y₂(t)")
    col1, col2 = st.columns(2)
    
    with col1:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=t*1000, y=y2, mode='lines', name='y₂(t)',
                                  line=dict(color='green', width=2)))
        fig3.update_layout(
            title=f'y₂(t) = {Amp2}·cos(2π·{f2}·t)',
            xaxis_title='Tiempo (ms)',
            yaxis_title='Amplitud',
            height=400,
            xaxis_range=[0, 6],
            hovermode='x unified'
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        Y2_fft = np.fft.fftshift(np.fft.fft(y2))
        Y2_mag = np.abs(Y2_fft) / N
        
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=fft_freqs, y=Y2_mag, mode='lines',
                                  line=dict(color='green', width=2)))
        fig4.update_layout(
            title='Espectro de y₂(t)',
            xaxis_title='Frecuencia (Hz)',
            yaxis_title='Amplitud',
            height=400,
            xaxis_range=[-2000, 2000],
            hovermode='x unified'
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Señal 3
    st.subheader("🔴 Señal y₃(t)")
    col1, col2 = st.columns(2)
    
    with col1:
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=t*1000, y=y3, mode='lines', name='y₃(t)',
                                  line=dict(color='red', width=2)))
        fig5.update_layout(
            title=f'y₃(t) = {Amp3}·cos(2π·{f3}·t)',
            xaxis_title='Tiempo (ms)',
            yaxis_title='Amplitud',
            height=400,
            xaxis_range=[0, 6],
            hovermode='x unified'
        )
        st.plotly_chart(fig5, use_container_width=True)
    
    with col2:
        Y3_fft = np.fft.fftshift(np.fft.fft(y3))
        Y3_mag = np.abs(Y3_fft) / N
        
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=fft_freqs, y=Y3_mag, mode='lines',
                                  line=dict(color='red', width=2)))
        fig6.update_layout(
            title='Espectro de y₃(t)',
            xaxis_title='Frecuencia (Hz)',
            yaxis_title='Amplitud',
            height=400,
            xaxis_range=[-2000, 2000],
            hovermode='x unified'
        )
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown("---")

    # ========== SEÑAL SUMA ==========
    st.header("➕ Señal Suma y(t)")
    st.subheader("y(t) = y₁(t) + y₂(t) + y₃(t)")
    
    # Mostrar solo el mínimo
    y_min = np.min(y_t)
    st.metric("📉 Mínimo de y(t)", f"{y_min:.4f}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig7 = go.Figure()
        fig7.add_trace(go.Scatter(x=t*1000, y=y_t, mode='lines', name='y(t)',
                                  line=dict(color='darkblue', width=2)))
        fig7.update_layout(
            title='Señal Suma y(t)',
            xaxis_title='Tiempo (ms)',
            yaxis_title='Amplitud',
            height=400,
            xaxis_range=[0, 6],
            hovermode='x unified'
        )
        st.plotly_chart(fig7, use_container_width=True)
    
    with col2:
        Y_fft = np.fft.fftshift(np.fft.fft(y_t))
        Y_mag = np.abs(Y_fft) / N
        
        fig8 = go.Figure()
        fig8.add_trace(go.Scatter(x=fft_freqs, y=Y_mag, mode='lines',
                                  line=dict(color='darkblue', width=2)))
        fig8.update_layout(
            title='Espectro de y(t)',
            xaxis_title='Frecuencia (Hz)',
            yaxis_title='Amplitud',
            height=400,
            xaxis_range=[-2000, 2000],
            hovermode='x unified'
        )
        st.plotly_chart(fig8, use_container_width=True)

    st.markdown("---")

    # ========== PORTADORA ==========
    st.header("📻 Señal Portadora")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig9 = go.Figure()
        fig9.add_trace(go.Scatter(x=t*1000, y=portadora, mode='lines', name='Portadora',
                                  line=dict(color='orange', width=1.5)))
        fig9.update_layout(
            title=f'Portadora: cos(2π·{f_port}·t)',
            xaxis_title='Tiempo (ms)',
            yaxis_title='Amplitud',
            height=400,
            xaxis_range=[0, 1],
            hovermode='x unified'
        )
        st.plotly_chart(fig9, use_container_width=True)
    
    with col2:
        Port_fft = np.fft.fftshift(np.fft.fft(portadora))
        Port_mag = np.abs(Port_fft) / N
        
        fig10 = go.Figure()
        fig10.add_trace(go.Scatter(x=fft_freqs, y=Port_mag, mode='lines',
                                   line=dict(color='orange', width=2)))
        fig10.update_layout(
            title='Espectro de la Portadora',
            xaxis_title='Frecuencia (Hz)',
            yaxis_title='Amplitud',
            height=400,
            xaxis_range=[-20000, 20000],
            hovermode='x unified'
        )
        st.plotly_chart(fig10, use_container_width=True)

    st.markdown("---")

    # ========== MODULACIÓN AM (DSB-LC) ==========
    st.header("📻 Modulación AM Convencional (DSB-LC)")
    
    # Usar la fórmula del código de referencia
    y_min_abs = np.abs(np.min(y_t))
    
    indices_mod = [1.2, 1.0, 0.7]
    mod_signals = []
    
    for mu in indices_mod:
        st.subheader(f"📊 Índice de modulación μ = {mu}")
        
        # Fórmula correcta del código de referencia
        a = y_min_abs / mu
        sig_lc = a * (1 + mu * y_t / y_min_abs) * portadora
        mod_signals.append(sig_lc)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfica temporal
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(x=t*1000, y=sig_lc, mode='lines',
                                         line=dict(color='red', width=1.5)))
            fig_time.update_layout(
                title=f'Señal AM con μ = {mu}',
                xaxis_title='Tiempo (ms)',
                yaxis_title='Amplitud',
                height=400,
                xaxis_range=[0, 5],
                hovermode='x unified'
            )
            st.plotly_chart(fig_time, use_container_width=True)
        
        with col2:
            # Espectro con FFT centrada
            fft_lc = np.fft.fftshift(np.fft.fft(sig_lc))
            fft_lc_mag = np.abs(fft_lc) / N
            
            fig_freq = go.Figure()
            fig_freq.add_trace(go.Scatter(x=fft_freqs, y=fft_lc_mag, mode='lines',
                                         line=dict(color='red', width=2)))
            fig_freq.update_layout(
                title=f'Espectro AM μ = {mu}',
                xaxis_title='Frecuencia (Hz)',
                yaxis_title='Amplitud',
                height=400,
                xaxis_range=[-20000, 20000],
                hovermode='x unified'
            )
            st.plotly_chart(fig_freq, use_container_width=True)
        
        st.markdown("---")

    # ========== RECTIFICACIÓN ==========
    st.header("🔧 Rectificación de las Señales Moduladas")
    
    for k, mu in enumerate(indices_mod):
        st.subheader(f"Señal Rectificada μ = {mu}")
        
        y_rect = np.abs(mod_signals[k])
        
        fig_rect = go.Figure()
        fig_rect.add_trace(go.Scatter(x=t*1000, y=y_rect, mode='lines',
                                     line=dict(color='darkorange', width=1.5)))
        fig_rect.update_layout(
            title=f'Señal AM Rectificada (μ = {mu})',
            xaxis_title='Tiempo (ms)',
            yaxis_title='Amplitud',
            height=400,
            xaxis_range=[0, 5],
            hovermode='x unified'
        )
        st.plotly_chart(fig_rect, use_container_width=True)

    st.success("✅ ¡Análisis completado!")