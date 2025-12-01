import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import io
import base64

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
    st.header(" Punto 1: Análisis de Series de Fourier")
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
    st.subheader("Señal Original")
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
    st.sidebar.subheader(" Análisis de Fourier")
    N = st.sidebar.slider("Número de armónicos (N):", min_value=1, max_value=50, value=10, step=1)

    c_n, n_values, a0, an_list, bn_list = coef_func(N)

    # Espectro en línea
    st.subheader(" Espectro en Línea")
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

    # Reconstrucción
    st.subheader(" Señal Reconstruida")
    fig_recon, ax_recon = plt.subplots(figsize=(14, 5))
    delta = 0.01
    ti = -T
    tf = T
    tiempo = np.arange(ti, tf + delta, delta)

    if tipo_senal == "Función definida en [-1,1] (Ej. 3.6.4)":
        y_reconstruida = reconstruir_senal(an_list, bn_list, tiempo, a0, usar_pi=True)
    else:
        y_reconstruida = reconstruir_senal(an_list, bn_list, tiempo, a0, usar_pi=False)

    y_original_recon = senal_func(tiempo, T)
    ax_recon.plot(tiempo, y_original_recon, 'b-', linewidth=2.5, label='Señal Original', alpha=0.7)
    ax_recon.plot(tiempo, y_reconstruida, 'r--', linewidth=2, label=f'Reconstrucción (N = {N})')
    ax_recon.set_xlabel('Tiempo (t)', fontsize=12)
    ax_recon.set_ylabel('x(t)', fontsize=12)
    ax_recon.set_title(f'{tipo_senal} - Señal Reconstruida', fontsize=14, fontweight='bold')
    ax_recon.legend(fontsize=11, loc='best')
    ax_recon.grid(True, alpha=0.3)
    ax_recon.axhline(y=0, color='k', linewidth=0.5)
    ax_recon.axvline(x=0, color='k', linewidth=0.5)
    ax_recon.axvline(x=-T, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    ax_recon.axvline(x=0, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    ax_recon.axvline(x=T, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    st.pyplot(fig_recon)

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
    st.header(" Punto 2: Modulación y Demodulación con Detección Sincrónica")
    st.markdown("**Implementación según Figuras 1 y 2 del laboratorio**")
    
    # Configuración de parámetros
    st.sidebar.header("Parámetros de Configuración")
    
    # Parámetros de la portadora
    st.sidebar.subheader("Señal Portadora")
    Ac = st.sidebar.slider("Amplitud de portadora (Ac)", 0.5, 2.0, 1.0, 0.1)
    fc = st.sidebar.slider("Frecuencia portadora fc (Hz)", 5000, 20000, 10000, 1000)
    
    # Parámetro del filtro pasa bajas
    st.sidebar.subheader("Filtro Pasa Bajas")
    cutoff = st.sidebar.slider("Frecuencia de corte del FPB (Hz)", 1000, 8000, 5000, 500)
    
    # Variables para almacenar las señales
    x_t = None
    fs = None
    
    st.sidebar.subheader("Cargar archivo de audio")
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
        st.header(" 1. Análisis de la Señal de Audio Original x(t)")
        
        # Calcular FFT
        x_f = np.fft.fft(x_t_raw)
        x_fcent = np.fft.fftshift(x_f)
        delta_f = 1 / (n * ts)
        f = np.arange(-n/2, n/2) * delta_f
        
        # Magnitud del espectro
        dep_original = np.abs(x_fcent / n)
        
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
            ax.set_ylabel('|X(ω)|')
            ax.set_title('Espectro de x(t) - Magnitud')
            ax.set_xlim([-fs/2, fs/2])
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
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
        
        st.markdown(audio_player(x_t_raw, fs, "🎵 Audio Original x(t)"), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ========== FILTRADO DE LA SEÑAL ==========
        st.header(" 2. Filtrado Pasa Bajas - Limitación de Ancho de Banda")
        
        st.info(f" **Frecuencia de corte seleccionada:** {cutoff} Hz")
        
        # Crear filtro pasa bajas ideal
        fpb = np.abs(f) <= cutoff
        
        # Aplicar filtro en dominio de frecuencia
        x_f_fil = x_fcent * fpb
        dep_filtrada = np.abs(x_f_fil / n)
        
        # Regresar al dominio del tiempo
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
            ax.set_ylabel('|X(ω)|')
            ax.set_title('Comparación de Espectros')
            ax.set_xlim([-10000, 10000])
            ax.legend()
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
            ax.set_title('Señal Filtrada x(t) en el Tiempo')
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
        st.header(" 3. Proceso de Modulación (Figura 1)")
        
        # Generar portadora
        carrier_cos = Ac * np.cos(2 * np.pi * fc * t)
        
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
            ax.set_ylabel('|X(ω)|')
            ax.set_title('X(ω) - Espectro de x(t)')
            ax.set_xlim([-10000, 10000])
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
            ax.set_ylabel('|Y(ω)|')
            ax.set_title('Y(ω) - Espectro de la señal modulada')
            ax.axvline(fc, color='r', linestyle='--', alpha=0.5, label=f'fc = {fc} Hz')
            ax.axvline(-fc, color='r', linestyle='--', alpha=0.5, label=f'-fc = {-fc} Hz')
            ax.set_xlim([-fs/2, fs/2])
            ax.legend()
            ax.grid(True, alpha=0.3)
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
            ax.set_ylabel("|X'(ω)|")
            ax.set_title("X'(ω) - Espectro antes del filtro")
            ax.axvline(2*fc, color='r', linestyle='--', alpha=0.5, label=f'2fc = {2*fc} Hz')
            ax.axvline(-2*fc, color='r', linestyle='--', alpha=0.5, label=f'-2fc = {-2*fc} Hz')
            ax.axvline(cutoff, color='orange', linestyle='--', alpha=0.5, label=f'Corte = {cutoff} Hz')
            ax.axvline(-cutoff, color='orange', linestyle='--', alpha=0.5)
            ax.set_xlim([-fs/2, fs/2])
            ax.legend()
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
        
        st.subheader("🔴 Punto 4: (1/2)x(t) recuperada - Después del FPB")
        
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
            ax.set_ylabel('|X(ω)|')
            ax.set_title('Espectro Comparativo')
            ax.set_xlim([-10000, 10000])
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")
        
        # ========== COMPARACIÓN FINAL ==========
        st.header(" 5. Comparación Final y Reproducción de Audio")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(audio_player(x_t, fs, "🎵 Señal Original x(t) (filtrada)"), unsafe_allow_html=True)
        
        with col2:
            st.markdown(audio_player(x_recovered, fs, "🎵 Señal Recuperada"), unsafe_allow_html=True)
        
        st.markdown(audio_player(y_t, fs, "📡 Señal Modulada y(t)"), unsafe_allow_html=True)
        
        st.markdown("""
        ### 📝 Instrucciones de Uso:
        
        1. **Cargar Audio:** Sube un archivo WAV en el panel lateral
        2. **Análisis Inicial:** Observa la señal original en tiempo y frecuencia
        3. **Ajustar Filtro:** Selecciona la frecuencia de corte del FPB para limitar el ancho de banda
        4. **Configurar Portadora:** Ajusta la amplitud y frecuencia de la señal portadora
        5. **Escuchar Resultados:** Compara el audio original con el recuperado
        """)
    
    else:
        st.info("👈 Por favor, carga un archivo de audio WAV desde el panel lateral para comenzar.")

# ============================================================================
# PUNTO 3: MODULACIÓN Y DEMODULACIÓN EN CUADRATURA DE FASE
# ============================================================================

elif pagina == "Punto 3: Modulación y Demodulación en cuadratura de fase":
    st.header(" Punto 3: Multiplexación en Cuadratura (Figura 3)")
    st.markdown("**Transmisión simultánea de dos señales usando ortogonalidad de senos y cosenos**")
    
    # Configuración de parámetros
    st.sidebar.header("Parámetros de Configuración")
    
    # Función para generar HTML de audio
    def audio_player(audio_data, sample_rate, label):
        if np.iscomplexobj(audio_data):
            audio_data = np.real(audio_data)
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
    
    # Cargar archivos de audio
    st.sidebar.subheader("Cargar archivos de audio")
    audio_file1 = st.sidebar.file_uploader("Señal x₁(t) (WAV)", type=['wav'], key='audio1_p3')
    audio_file2 = st.sidebar.file_uploader("Señal x₂(t) (WAV)", type=['wav'], key='audio2_p3')
    
    # Variables globales
    x1_t = None
    x2_t = None
    fs = None
    n = None
    dur_aud = None
    ts = None
    t = None
    
    if audio_file1 is not None and audio_file2 is not None:
        # Leer primer archivo
        fs1, x1_t = wavfile.read(audio_file1)
        if x1_t.ndim == 2:
            x1_t = np.mean(x1_t, axis=1)
        x1_t = x1_t.astype(float) / np.max(np.abs(x1_t))
        
        # Leer segundo archivo
        fs2, x2_t = wavfile.read(audio_file2)
        if x2_t.ndim == 2:
            x2_t = np.mean(x2_t, axis=1)
        x2_t = x2_t.astype(float) / np.max(np.abs(x2_t))
        
        # Usar misma frecuencia de muestreo
        fs = min(fs1, fs2)
        
        # Limitar duración a 10 segundos y ajustar a la misma longitud
        max_samples = min(len(x1_t), len(x2_t), int(10 * fs))
        x1_t = x1_t[:max_samples]
        x2_t = x2_t[:max_samples]
        
        n = len(x1_t)
        dur_aud = n / fs
        ts = 1 / fs
        t = np.arange(n) * ts
        
        st.success(f"✅ Archivos cargados correctamente")
        st.info(f" Frecuencia de muestreo: {fs} Hz | Duración: {dur_aud:.2f} s | Muestras: {n}")
        
        # Parámetros de la portadora
        st.sidebar.subheader("Parámetros Portadora")
        fc = st.sidebar.slider("Frecuencia portadora fc (Hz)", 10000, 50000, 20000, 1000)
        fc_filter = st.sidebar.slider("Frecuencia de corte FPB (Hz)", 500, 10000, 5000, 100)
        
        # Configuración del eje de frecuencia
        delta_f = 1 / (n * ts)
        f = np.arange(-n/2, n/2) * delta_f
        
        # ========== SEÑALES ORIGINALES ==========
        st.header(" Señales Moduladoras Originales")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Señal x₁(t)")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
            
            # Tiempo
            ax1.plot(t, x1_t, 'b', linewidth=1)
            ax1.set_xlabel('Tiempo (s)')
            ax1.set_ylabel('Amplitud')
            ax1.set_title('x₁(t) en el tiempo')
            ax1.grid(True, alpha=0.3)
            
            # Frecuencia
            x1_f = np.fft.fft(x1_t)
            x1_fcent = np.fft.fftshift(x1_f)
            ax2.plot(f, np.abs(x1_fcent/n), 'b', linewidth=1)
            ax2.set_xlabel('Frecuencia (Hz)')
            ax2.set_ylabel('Magnitud normalizada')
            ax2.set_title('Espectro de x₁(t)')
            ax2.set_xlim([-5000, 5000])
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            st.markdown(audio_player(x1_t, fs, "Audio x₁(t)"), unsafe_allow_html=True)
        
        with col2:
            st.subheader("Señal x₂(t)")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
            
            # Tiempo
            ax1.plot(t, x2_t, 'r', linewidth=1)
            ax1.set_xlabel('Tiempo (s)')
            ax1.set_ylabel('Amplitud')
            ax1.set_title('x₂(t) en el tiempo')
            ax1.grid(True, alpha=0.3)
            
            # Frecuencia
            x2_f = np.fft.fft(x2_t)
            x2_fcent = np.fft.fftshift(x2_f)
            ax2.plot(f, np.abs(x2_fcent/n), 'r', linewidth=1)
            ax2.set_xlabel('Frecuencia (Hz)')
            ax2.set_ylabel('Magnitud normalizada')
            ax2.set_title('Espectro de x₂(t)')
            ax2.set_xlim([-5000, 5000])
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            st.markdown(audio_player(x2_t, fs, "Audio x₂(t)"), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ========== MODULACIÓN ==========
        st.header(" MODULACIÓN - Lado Izquierdo de la Figura 3")
        
        # Generar portadoras en cuadratura
        cos_carrier = np.cos(2 * np.pi * fc * t)
        sin_carrier = np.sin(2 * np.pi * fc * t)
        
        # 🔴 Punto Rojo 1: x₁(t) * cos(ωc·t)
        y1_t = x1_t * cos_carrier
        
        st.subheader("🔴 Punto Rojo 1: y₁(t) = x₁(t)·cos(ωc·t)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 3))
            t_display = min(0.01, dur_aud)
            idx_display = int(t_display * fs)
            ax.plot(t[:idx_display], y1_t[:idx_display], 'b', linewidth=1)
            ax.set_xlabel('Tiempo (s)')
            ax.set_ylabel('Amplitud')
            ax.set_title('y₁(t) en el tiempo')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            y1_f = np.fft.fft(y1_t)
            y1_fcent = np.fft.fftshift(y1_f)
            
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(f, np.abs(y1_fcent/n), 'b', linewidth=1)
            ax.set_xlabel('Frecuencia (Hz)')
            ax.set_ylabel('Magnitud normalizada')
            ax.set_title('Espectro de y₁(t)')
            ax.axvline(fc, color='red', linestyle='--', alpha=0.5)
            ax.axvline(-fc, color='red', linestyle='--', alpha=0.5)
            ax.set_xlim([-fc-10000, fc+10000])
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # 🔴 Punto Rojo 2: x₂(t) * sin(ωc·t)
        y2_t = x2_t * sin_carrier
        
        st.subheader("🔴 Punto Rojo 2: y₂(t) = x₂(t)·sin(ωc·t)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(t[:idx_display], y2_t[:idx_display], 'r', linewidth=1)
            ax.set_xlabel('Tiempo (s)')
            ax.set_ylabel('Amplitud')
            ax.set_title('y₂(t) en el tiempo')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            y2_f = np.fft.fft(y2_t)
            y2_fcent = np.fft.fftshift(y2_f)
            
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(f, np.abs(y2_fcent/n), 'r', linewidth=1)
            ax.set_xlabel('Frecuencia (Hz)')
            ax.set_ylabel('Magnitud normalizada')
            ax.set_title('Espectro de y₂(t)')
            ax.axvline(fc, color='red', linestyle='--', alpha=0.5)
            ax.axvline(-fc, color='red', linestyle='--', alpha=0.5)
            ax.set_xlim([-fc-10000, fc+10000])
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # 🔴 Punto Rojo 3: Suma Σ
        y_sum_t = y1_t + y2_t
        
        st.subheader("🔴 Punto Rojo 3: Señal Transmitida = y₁(t) + y₂(t)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(t[:idx_display], y_sum_t[:idx_display], 'purple', linewidth=1)
            ax.set_xlabel('Tiempo (s)')
            ax.set_ylabel('Amplitud')
            ax.set_title('Señal transmitida en el tiempo')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            ysum_f = np.fft.fft(y_sum_t)
            ysum_fcent = np.fft.fftshift(ysum_f)
            
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(f, np.abs(ysum_fcent/n), 'purple', linewidth=1)
            ax.set_xlabel('Frecuencia (Hz)')
            ax.set_ylabel('Magnitud normalizada')
            ax.set_title('Espectro de la señal transmitida')
            ax.axvline(fc, color='red', linestyle='--', alpha=0.5)
            ax.axvline(-fc, color='red', linestyle='--', alpha=0.5)
            ax.set_xlim([-fc-10000, fc+10000])
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        st.markdown(audio_player(y_sum_t, fs, "Señal Transmitida"), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ========== DEMODULACIÓN ==========
        st.header(" DEMODULACIÓN - Lado Derecho de la Figura 3")
        
        st.markdown("""
        **Proceso de recuperación usando ortogonalidad:**
        - Para recuperar x₁(t): multiplicar por cos(ωc·t) y aplicar FPB
        - Para recuperar x₂(t): multiplicar por sin(ωc·t) y aplicar FPB
        """)
        
        # Filtro pasa-bajas ideal
        fpb = np.abs(f) <= fc_filter
        
        # ========== RECUPERACIÓN DE x₁(t) ==========
        st.subheader(" Recuperación de x₁(t)")
        
        # 🔴 Multiplicar señal recibida por cos(ωc·t)
        x1_prime_t = y_sum_t * cos_carrier
        
        st.markdown("🔴 Después de multiplicar por cos(ωc·t)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(t[:idx_display], x1_prime_t[:idx_display], 'cyan', linewidth=1)
            ax.set_xlabel('Tiempo (s)')
            ax.set_ylabel('Amplitud')
            ax.set_title("x₁'(t) antes del filtro")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            x1p_f = np.fft.fft(x1_prime_t)
            x1p_fcent = np.fft.fftshift(x1p_f)
            
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(f, np.abs(x1p_fcent/n), 'cyan', linewidth=1)
            ax.set_xlabel('Frecuencia (Hz)')
            ax.set_ylabel('Magnitud normalizada')
            ax.set_title("Espectro de x₁'(t)")
            ax.set_xlim([-2*fc-10000, 2*fc+10000])
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # 🔴 Aplicar FPB y recuperar
        x1p_f_fil = x1p_fcent * fpb
        x1p_f_filco = np.fft.ifftshift(x1p_f_fil)
        x1_rec_t = np.real(np.fft.ifft(x1p_f_filco)) * 2
        
        st.markdown("🔴 Después del FPB - Señal x₁(t) recuperada")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(t, x1_rec_t, 'g', linewidth=1.5, label='Recuperada')
            ax.plot(t, x1_t, 'b--', alpha=0.5, linewidth=1, label='Original')
            ax.set_xlabel('Tiempo (s)')
            ax.set_ylabel('Amplitud')
            ax.set_title('x₁(t) recuperada vs original')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            x1rec_f = np.fft.fft(x1_rec_t)
            x1rec_fcent = np.fft.fftshift(x1rec_f)
            
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(f, np.abs(x1rec_fcent/n), 'g', linewidth=1.5, label='Recuperada')
            ax.plot(f, np.abs(x1_fcent/n), 'b--', alpha=0.5, linewidth=1, label='Original')
            ax.set_xlabel('Frecuencia (Hz)')
            ax.set_ylabel('Magnitud normalizada')
            ax.set_title('Espectro comparativo')
            ax.set_xlim([-10000, 10000])
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        st.markdown(audio_player(x1_rec_t, fs, "x₁(t) Recuperada"), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ========== RECUPERACIÓN DE x₂(t) ==========
        st.subheader("🔴 Recuperación de x₂(t)")
        
        # 🔴 Multiplicar señal recibida por sin(ωc·t)
        x2_prime_t = y_sum_t * sin_carrier
        
        st.markdown("🔴 Después de multiplicar por sin(ωc·t)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(t[:idx_display], x2_prime_t[:idx_display], 'orange', linewidth=1)
            ax.set_xlabel('Tiempo (s)')
            ax.set_ylabel('Amplitud')
            ax.set_title("x₂'(t) antes del filtro")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            x2p_f = np.fft.fft(x2_prime_t)
            x2p_fcent = np.fft.fftshift(x2p_f)
            
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(f, np.abs(x2p_fcent/n), 'orange', linewidth=1)
            ax.set_xlabel('Frecuencia (Hz)')
            ax.set_ylabel('Magnitud normalizada')
            ax.set_title("Espectro de x₂'(t)")
            ax.set_xlim([-2*fc-10000, 2*fc+10000])
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # 🔴 Aplicar FPB y recuperar
        x2p_f_fil = x2p_fcent * fpb
        x2p_f_filco = np.fft.ifftshift(x2p_f_fil)
        x2_rec_t = np.real(np.fft.ifft(x2p_f_filco)) * 2
        
        st.markdown("🔴 Después del FPB - Señal x₂(t) recuperada")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(t, x2_rec_t, 'g', linewidth=1.5, label='Recuperada')
            ax.plot(t, x2_t, 'r--', alpha=0.5, linewidth=1, label='Original')
            ax.set_xlabel('Tiempo (s)')
            ax.set_ylabel('Amplitud')
            ax.set_title('x₂(t) recuperada vs original')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            x2rec_f = np.fft.fft(x2_rec_t)
            x2rec_fcent = np.fft.fftshift(x2rec_f)
            
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(f, np.abs(x2rec_fcent/n), 'g', linewidth=1.5, label='Recuperada')
            ax.plot(f, np.abs(x2_fcent/n), 'r--', alpha=0.5, linewidth=1, label='Original')
            ax.set_xlabel('Frecuencia (Hz)')
            ax.set_ylabel('Magnitud normalizada')
            ax.set_title('Espectro comparativo')
            ax.set_xlim([-10000, 10000])
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        st.markdown(audio_player(x2_rec_t, fs, "x₂(t) Recuperada"), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ========== DEMOSTRACIÓN MATEMÁTICA ==========
        st.header(" Demostración Matemática")
        
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
        
        st.success("✅ El sistema de multiplexación en cuadratura permite transmitir dos señales diferentes simultáneamente en la misma frecuencia portadora, aprovechando la ortogonalidad entre cos(ωc·t) y sin(ωc·t).")
    
    else:
        st.info("👈 Por favor, carga dos archivos de audio WAV desde el panel lateral para comenzar.")
        st.markdown("""
        ### 📝 Instrucciones:
        
        1. **Cargar dos audios:** Necesitas dos archivos WAV diferentes para x₁(t) y x₂(t)
        2. **Configurar parámetros:** Ajusta la frecuencia portadora y el filtro pasa bajas
        3. **Observar modulación:** Verás cómo se combinan las dos señales
        4. **Verificar demodulación:** Comprueba que ambas señales se recuperan correctamente
        
        **Concepto clave:** Este sistema aprovecha la ortogonalidad matemática entre cos(ωc·t) y sin(ωc·t) 
        para transmitir dos señales independientes en la misma frecuencia portadora.
        """)

elif pagina == "Punto 4: Modulación de amplitud DSB-LC": 
    st.header("Punto 4: Modulación de Amplitud DSB-LC")

    # ================== PUNTO 4 ==================
    st.title("Modulación de Señales Sinusoidales")
    st.write("Configura los parámetros de las señales y visualiza la modulación AM")

    # Sidebar para inputs
    st.sidebar.header("Parámetros de las Señales")

    st.sidebar.subheader("Señal 1")
    f1 = st.sidebar.number_input("Frecuencia f1 (Hz)", min_value=1.0, max_value=5000.0, value=250.0, step=10.0)
    Amp1 = st.sidebar.number_input("Amplitud A1", min_value=0.1, max_value=5.0, value=1.2, step=0.1)

    st.sidebar.subheader("Señal 2")
    f2 = st.sidebar.number_input("Frecuencia f2 (Hz)", min_value=1.0, max_value=5000.0, value=500.0, step=10.0)
    Amp2 = st.sidebar.number_input("Amplitud A2", min_value=0.1, max_value=5.0, value=0.8, step=0.1)

    st.sidebar.subheader("Señal 3")
    f3 = st.sidebar.number_input("Frecuencia f3 (Hz)", min_value=1.0, max_value=5000.0, value=1000.0, step=10.0)
    Amp3 = st.sidebar.number_input("Amplitud A3", min_value=0.1, max_value=5.0, value=0.4, step=0.1)

    st.sidebar.subheader("Portadora")
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

    # Botón
    if st.button("Generar Análisis Completo"):
        
        # 1. Señal suma
        st.header("1. Señal Suma y(t)")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(t * 1000, y_t, color='darkblue')
        ax1.set_xlabel('Tiempo [ms]')
        ax1.set_ylabel('Amplitud')
        ax1.set_title(r'Señal suma $y(t) = A y_1 + B y_2 + C y_3$')
        ax1.set_xlim(0, 6)
        ax1.grid(True, which='both', linestyle='--')
        st.pyplot(fig1)
        
        # FFT señal suma
        Y_fft = np.fft.fft(y_t)
        frq = np.fft.fftfreq(N, d=1/fs)
        idx_pos = np.where(frq >= 0)
        f_axis = frq[idx_pos]
        Y_pos = Y_fft[idx_pos]
        DEP_y = (np.abs(Y_pos)**2) / (N**2)
        DEP_norm = DEP_y / np.max(DEP_y)
        
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.stem(f_axis, DEP_norm, linefmt='g-', markerfmt='go')
        ax2.set_xlabel('Frecuencia [Hz]')
        ax2.set_ylabel('Magnitud Normalizada (DEP)')
        ax2.set_title(r'Densidad Espectral de Potencia de $y(t)$')
        ax2.set_xlim(0, 1500)
        ax2.grid(True)
        st.pyplot(fig2)
        
        # 2. Modulación DSB-SC
        st.header("2. Modulación DSB-SC")
        portadora = np.cos(w_port * t)
        y_mod_sc = y_t * portadora
        
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(t*1000, y_mod_sc, color='purple')
        ax3.set_xlabel('Tiempo [ms]')
        ax3.set_ylabel('Amplitud')
        ax3.set_title('Señal Modulada DSB-SC')
        ax3.set_xlim(0, 3)
        ax3.grid(True)
        st.pyplot(fig3)
        
        Spec_sc = np.fft.fft(y_mod_sc)
        Spec_sc_pos = Spec_sc[idx_pos]
        Pwr_sc = (np.abs(Spec_sc_pos)**2) / (N**2)
        Pwr_sc_norm = Pwr_sc / np.max(Pwr_sc)
        
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        ax4.stem(f_axis, Pwr_sc_norm, linefmt='purple', markerfmt='mo')
        ax4.set_xlabel('Freq [Hz]')
        ax4.set_ylabel('Magnitud Normalizada')
        ax4.set_title('Espectro de la señal DSB-SC')
        ax4.set_xlim(0, 24000)
        ax4.grid(True)
        st.pyplot(fig4)
        
        # 3. Modulación DSB-LC
        st.header("3. Modulación AM Convencional (DSB-LC)")
        amp_peak = np.max(np.abs(y_t))
        y_unit = y_t / amp_peak
        
        st.write(f"Amplitud pico: {amp_peak:.4f}")
        
        indices_mod = [1.2, 1.0, 0.7]
        mod_signals = []
        
        for idx in indices_mod:
            sig_lc = (1 + idx * y_unit) * portadora
            mod_signals.append(sig_lc)
            
            st.subheader(f"Índice de modulación μ = {idx}")
            
            fig5, ax5 = plt.subplots(figsize=(10, 4))
            ax5.plot(t*1000, sig_lc, color='tab:red')
            ax5.set_xlim(0, 5)
            ax5.set_xlabel("Tiempo [ms]")
            ax5.set_title(f"AM DSB-LC μ = {idx}")
            ax5.grid(True)
            st.pyplot(fig5)
            
            fft_lc = np.fft.fft(sig_lc)
            fft_lc_pos = fft_lc[idx_pos]
            Pwr_lc = (np.abs(fft_lc_pos)**2)/(N**2)
            Pwr_lc_norm = Pwr_lc/np.max(Pwr_lc)
            
            fig6, ax6 = plt.subplots(figsize=(10, 4))
            ax6.stem(f_axis, Pwr_lc_norm, linefmt='r-', markerfmt='ro')
            ax6.set_xlim(0, 25000)
            ax6.set_xlabel("Frecuencia [Hz]")
            ax6.set_title(f"Espectro AM μ = {idx}")
            ax6.grid(True)
            st.pyplot(fig6)
        
        # 4. Rectificación
        st.header("4. Rectificación de las Señales")
        
        for k, idx in enumerate(indices_mod):
            y_rect = np.abs(mod_signals[k])
            st.subheader(f"Rectificada μ = {idx}")
            
            fig7, ax7 = plt.subplots(figsize=(10, 4))
            ax7.plot(t*1000, y_rect, color='tab:orange')
            ax7.set_xlim(0, 5)
            ax7.grid(True)
            st.pyplot(fig7)
        
        st.success("¡Análisis completado!")


