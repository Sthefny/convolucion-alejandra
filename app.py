import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# ===========================
# CONFIGURACIÓN GLOBAL
# ===========================
st.set_page_config(page_title="Convolución Interactiva", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f6f8fb 0%, #eef6ff 50%, #f7f3ff 100%);
    color: #0f1724;
    font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.glass {
    background: rgba(255, 255, 255, 0.55);
    border-radius: 14px;
    box-shadow: 0 6px 18px rgba(16, 24, 40, 0.06);
    padding: 14px;
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    border: 1px solid rgba(255,255,255,0.6);
}
.big-title {
    font-size: 26px;
    font-weight: 700;
    background: linear-gradient(90deg, #06b6d4, #8b5cf6);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
.stButton>button {
    background: linear-gradient(90deg,#06b6d4,#7c3aed);
    color: white;
    border-radius: 8px;
    padding: 6px 12px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ===========================
# SIDEBAR PRINCIPAL
# ===========================
st.sidebar.title("⚙️ Navegación")
punto = st.sidebar.selectbox(
    "Selecciona el punto:",
    ["Punto 1 — Convolución Discreta", 
     "Punto 1 — Convolución Continua", 
     "Punto 2: Comparación convoluciones continuas"]
)

# --- Información del laboratorio ---
st.sidebar.markdown("---")  # línea divisoria
st.sidebar.subheader("Laboratorio 2 Señales y Sistemas")
st.sidebar.markdown(
    """
**Tema:** Convolución Continua y Discreta  
  

**Integrantes:**  
-  Alejandra Tuirán  
-  Sebastian Lozano
-  Valentinca Charris 
"""
)


# ===========================
# PARTE A — CONVOLUCIÓN DISCRETA
# ===========================
if punto == "Punto 1 — Convolución Discreta":
    import numpy as np
    import matplotlib.pyplot as plt
    import time

    # --- Señales ---
    def triangular_signal(n):
        return np.where(np.abs(n) < 6, 6 - np.abs(n), 0)

    def rectangular_signal(n):
        return np.where((n >= -5) & (n < 5), 1, 0)

    def u(n):
        return np.where(n >= 0, 1, 0)

    # Convolución 1
    n1 = np.arange(-10, 11)
    x1 = triangular_signal(n1)
    h1 = rectangular_signal(n1)

    # Convolución 2
    n2 = np.arange(-10, 20)
    x2 = u(n2 + 3) - u(n2 - 7)
    h2 = ((6/7)**n2) * (u(n2) - u(n2 - 9))
    h2 = np.where(n2 >= 0, h2, 0)

    # --- Funciones auxiliares ---
    def draw_signals(x, h, n_axis, title_x, title_h):
        fig, axs = plt.subplots(1, 3, figsize=(12, 3))
        fig.suptitle("Visualización de señales", fontsize=14, fontweight='bold')

        axs[0].stem(n_axis, x, basefmt=" ")
        axs[0].set_title(title_x)
        axs[0].grid(True, linestyle='--', alpha=0.4)

        axs[1].stem(n_axis, h, basefmt=" ")
        axs[1].set_title(title_h)
        axs[1].grid(True, linestyle='--', alpha=0.4)

        axs[2].stem(-n_axis, h[::-1], basefmt=" ")
        axs[2].set_title(r"$h[-n]$")
        axs[2].grid(True, linestyle='--', alpha=0.4)

        fig.tight_layout()
        return fig

    def draw_frame_single_figure(x, h, step, y_display, ny, n_axis):
        lx = len(x)
        lh = len(h)
        pad = lh - 1
        x_k = np.concatenate((np.zeros(pad), x, np.zeros(pad)))
        eje_k = np.arange(n_axis[0] - pad, n_axis[-1] + pad + 1)

        N = lx + lh - 1
        h_rev = h[::-1]
        h_nk = np.zeros_like(x_k)
        start = step
        h_nk[start:start+lh] = h_rev

        v_k = x_k * h_nk

        fig, axs = plt.subplots(2, 2, figsize=(10, 6))
        fig.suptitle(f"Convolución — Paso {step+1}/{N}", fontsize=14, fontweight='bold')

        lim_min = eje_k[0] - 1
        lim_max = eje_k[-1] + 1

        axs[0,0].stem(eje_k, x_k, basefmt=" ")
        axs[0,0].set_title("x[k]")
        axs[0,0].set_xlim(lim_min, lim_max)
        axs[0,0].grid(True, linestyle='--', alpha=0.4)

        axs[0,1].stem(eje_k, h_nk, linefmt='C1-', markerfmt='C1o', basefmt=" ")
        axs[0,1].set_title("h[n−k]")
        axs[0,1].set_xlim(lim_min, lim_max)
        axs[0,1].grid(True, linestyle='--', alpha=0.4)

        axs[1,0].stem(eje_k, v_k, linefmt='C2-', markerfmt='C2o', basefmt=" ")
        axs[1,0].set_title("v[k] = x[k]·h[n−k]")
        axs[1,0].set_xlim(lim_min, lim_max)
        axs[1,0].grid(True, linestyle='--', alpha=0.4)

        axs[1,1].stem(ny[:step+1], y_display[:step+1], linefmt='C3-', markerfmt='C3o', basefmt=" ")
        axs[1,1].set_title("y[n] parcial (construyéndose)")
        axs[1,1].set_xlim(ny[0] - 1, ny[-1] + 1)
        axs[1,1].grid(True, linestyle='--', alpha=0.4)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    def run_animation(x, h, n_axis, frame_delay=0.7):
        lx, lh = len(x), len(h)
        pad = lh - 1
        x_k = np.concatenate((np.zeros(pad), x, np.zeros(pad)))
        N = lx + lh - 1
        ny = np.arange(2 * n_axis[0], 2 * n_axis[0] + N)

        for step in range(N):
            y_full = np.zeros(N)
            for s in range(step + 1):
                h_rev = h[::-1]
                h_nk = np.zeros_like(x_k)
                h_nk[s:s+lh] = h_rev
                v_k = x_k * h_nk
                y_full[s] = np.sum(v_k)

            y_display = np.zeros(N)
            y_display[:step+1] = y_full[:step+1]

            fig = draw_frame_single_figure(x, h, step, y_display, ny, n_axis)
            visor.pyplot(fig, clear_figure=True)
            plt.close(fig)
            time.sleep(frame_delay)

        y_complete = np.convolve(x, h)
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 3))
        ax2.stem(ny, y_complete, linefmt='C3-', markerfmt='C3o', basefmt=" ")
        ax2.set_title("y[n] — Resultado final")
        ax2.set_xlim(ny[0] - 1, ny[-1] + 1)
        ax2.grid(True, linestyle='--', alpha=0.4)
        fig2.tight_layout()
        visor.pyplot(fig2, clear_figure=True)
        plt.close(fig2)

    # --- Interfaz ---
    st.markdown('<h1 class="big-title">🟦 Punto 1 — Convolución Discreta</h1>', unsafe_allow_html=True)

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    choice = st.selectbox("Selecciona la convolución:",
                          ("Convolución 1 ",
                           "Convolución 2 "))
    st.markdown('</div>', unsafe_allow_html=True)

    if choice.startswith("Convolución 1"):
        n_axis, x, h = n1, x1, h1
        title_x = r"$x[n] = 6 - |n|, |n| < 6$"
        title_h = r"$h[n] = u[n+5] - u[n-5]$"
    else:
        n_axis, x, h = n2, x2, h2
        title_x = r"$x[n] = u[n+3] - u[n-7]$"
        title_h = r"$h[n] = (6/7)^n (u[n] - u[n-9])$"

    st.subheader("Señales originales")
    st.pyplot(draw_signals(x, h, n_axis, title_x, title_h))

    visor = st.empty()
    if st.button("▶ Iniciar animación"):
 
        run_animation(x, h, n_axis, frame_delay=0.4)
elif punto == "Punto 1 — Convolución Continua":
    # ======================
    # Convolución continua (animación corregida y longitudes iguales)
    # ======================

    st.set_page_config(page_title="Convolución Continua — Animación", layout="wide")

    # --- Parámetros ---
    delta = 0.05       # paso temporal
    FRAME_DELAY = 0.0000000000000000000000000000000001  # segundos por frame

    # --- Definición de señales ---
    t_a = np.arange(-1, 5 + delta, delta)
    x_a = np.piecewise(t_a, [t_a < 0, (t_a >= 0) & (t_a < 3), (t_a >= 3) & (t_a < 5), t_a >= 5],
                       [0, 2, -2, 0])

    t_b = np.arange(-2, 2 + delta, delta)
    x_b = np.piecewise(t_b, [t_b < -1, (t_b >= -1) & (t_b <= 1), t_b > 1],
                       [0, lambda t: -t, 0])

    t_c = np.arange(-2, 5 + delta, delta)
    x_c = np.piecewise(
        t_c,
        [t_c < -1, (t_c >= -1) & (t_c < 1), (t_c >= 1) & (t_c < 3), (t_c >= 3) & (t_c < 5), t_c >= 5],
        [0, 2, lambda t: -2*(t-1)+2, -2, 0]
    )

    t_d = np.arange(-3, 3 + delta, delta)
    x_d = np.piecewise(t_d, [t_d < -3, (t_d >= -3) & (t_d <= 3), t_d > 3],
                       [0, lambda t: np.exp(-np.abs(t)), 0])

    signals = {"a": (t_a, x_a), "b": (t_b, x_b), "c": (t_c, x_c), "d": (t_d, x_d)}

    # --- Estado ---
    if "cont_anim_running" not in st.session_state:
        st.session_state.cont_anim_running = False
    if "cont_anim_stop" not in st.session_state:
        st.session_state.cont_anim_stop = False

    # --- Interfaz ---
    st.markdown("<h2>Convolución Continua — Paso a paso</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])

    with col1:
        s1 = st.selectbox("Primera señal x(t)", ["a", "b", "c", "d"], index=0)
        s2 = st.selectbox("Segunda señal h(t)", ["a", "b", "c", "d"], index=0)
        start_btn = st.button("▶ Iniciar animación continua")
        stop_btn = st.button("⏹ Detener animación")

    with col2:
        visor = st.empty()

    if stop_btn:
        st.session_state.cont_anim_stop = True

    # --- Obtener señales ---
    t_x, x_t = signals[s1]
    t_h, h_t = signals[s2]

    # --- Calcular convolución (numérica) ---
    y_conv = np.convolve(x_t, h_t, mode='full') * delta
    # Asegurar longitudes iguales
    t_conv = np.linspace(t_x[0] + t_h[0], t_x[-1] + t_h[-1], len(y_conv))

    # --- Vista previa ---
    fig_prev, axs_prev = plt.subplots(2, 1, figsize=(10, 5))
    fig_prev.suptitle("Vista previa de señales", fontsize=14, fontweight="bold")
    axs_prev[0].plot(t_x, x_t, linewidth=2, label="x(t)")
    axs_prev[0].legend(); axs_prev[0].grid(alpha=0.4)
    axs_prev[1].plot(t_h, h_t, linewidth=2, color="C1", label="h(t)")
    axs_prev[1].legend(); axs_prev[1].grid(alpha=0.4)
    fig_prev.tight_layout(rect=[0, 0.03, 1, 0.95])
    visor.pyplot(fig_prev, clear_figure=True)
    plt.close(fig_prev)

    # --- Animación ---
    def run_continuous_animation():
        st.session_state.cont_anim_stop = False
        st.session_state.cont_anim_running = True

        for k, t_k in enumerate(t_conv):
            if st.session_state.cont_anim_stop:
                st.session_state.cont_anim_running = False
                return

            # Desplazamiento coherente con la convolución continua
            h_shifted = np.interp(t_x, t_h + t_k, h_t, left=0, right=0)
            product = x_t * h_shifted

            # --- Gráficas ---
            fig, axs = plt.subplots(3, 1, figsize=(10, 7))
            fig.suptitle(f"t = {t_k:.2f} — Paso {k+1}/{len(t_conv)}", fontsize=14, fontweight="bold")

            axs[0].plot(t_x, x_t, label="x(t)", linewidth=2)
            axs[0].plot(t_x, h_shifted, label="h(t) desplazada", color="C1", linewidth=2)
            axs[0].legend(); axs[0].grid(alpha=0.4)

            axs[1].plot(t_x, product, color="C2")
            axs[1].set_title("Producto x(t)·h(t desplazada)")
            axs[1].grid(alpha=0.4)

            axs[2].plot(t_conv[:k+1], y_conv[:k+1], color="purple")
            axs[2].set_title("y(t) — construcción progresiva")
            axs[2].grid(alpha=0.4)

            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            visor.pyplot(fig, clear_figure=True)
            plt.close(fig)

            time.sleep(FRAME_DELAY)

        # --- Resultado final ---
        fig_final, ax_final = plt.subplots(1, 1, figsize=(10, 3))
        ax_final.plot(t_conv, y_conv, color="purple", linewidth=2)
        ax_final.set_title("y(t) — Convolución completa")
        ax_final.grid(alpha=0.4)
        fig_final.tight_layout()
        visor.pyplot(fig_final, clear_figure=True)
        plt.close(fig_final)

        st.session_state.cont_anim_running = False

    if start_btn and not st.session_state.cont_anim_running:
        run_continuous_animation()
elif punto == "Punto 2: Comparación convoluciones continuas":
    import numpy as np
    import matplotlib.pyplot as plt
    import streamlit as st

    st.markdown("<h2>Comparación entre convolución manual y np.convolve</h2>", unsafe_allow_html=True)

    delta = 0.05

    def u(t):
        return np.where(t >= 0, 1, 0)

    # --- Selección de caso ---
    caso = st.selectbox("Selecciona el caso", ["a", "b", "c"], index=0)
    visor = st.empty()

    # =============== CASO A ===============
    if caso == "a":
        t_x = np.arange(-1, 5 + delta, delta)
        t_h = np.arange(0, 6 + delta, delta)

        x_t = np.exp(-4 * t_x / 5) * (u(t_x + 1) - u(t_x - 5))
        h_t = np.exp(-t_h / 4) * u(t_h)

        # Convolución con numpy
        y_tpy = np.convolve(x_t, h_t) * delta
        len_y = len(t_x) + len(t_h) - 1
        t_y = np.arange(t_x[0] + t_h[0], t_x[0] + t_h[0] + len_y * delta, delta)

        # Convolución manual
        t_y1 = np.arange(-5, -1, delta)
        t_y2 = np.arange(-1, 5, delta)
        t_y3 = np.arange(5, 20, delta)
        y_t1 = np.zeros_like(t_y1)
        y_t2 = (20/11) * np.exp(-t_y2/4) * (np.exp(11/20) - np.exp(-11*t_y2/20))
        y_t3 = (20/11) * np.exp(-t_y3/4) * (np.exp(11/20) - np.exp(-11/4))
        y_tm = np.concatenate((y_t1, y_t2, y_t3))
        t_m = np.concatenate((t_y1, t_y2, t_y3))

    # =============== CASO B ===============
    elif caso == "b":
        t_b = np.arange(-1, 6 + delta, delta)
        t_hb = np.arange(-4, 4 + delta, delta)

        h_t = np.exp(-0.5 * t_b) * u(t_b + 1)
        x_t = np.exp(0.5 * t_hb) * (u(t_hb + 4) - u(t_hb)) + np.exp(-0.5 * t_hb) * (u(t_hb) - u(t_hb - 4))

        y_tpy = np.convolve(x_t, h_t) * delta
        len_yb = len(x_t) + len(h_t) - 1
        t_y = np.arange(t_hb[0] + t_b[0], t_hb[0] + t_b[0] + len_yb * delta, delta)

        # Manual
        tm1_b = np.arange(-6, -5, delta)
        tm2_b = np.arange(-5, -1, delta)
        tm3_b = np.arange(-1, 3, delta)
        tm4_b = np.arange(3, 10, delta)

        ym1_b = np.zeros_like(tm1_b)
        ym2_b = np.exp(1 + tm2_b/2) - np.exp(-4 - tm2_b/2)
        ym3_b = np.exp(-tm3_b/2) * (tm3_b + 2 - np.exp(-4))
        ym4_b = (5 - np.exp(-4)) * np.exp(-tm4_b/2)

        y_tm = np.concatenate((ym1_b, ym2_b, ym3_b, ym4_b))
        t_m = np.concatenate((tm1_b, tm2_b, tm3_b, tm4_b))

    # =============== CASO C ===============
    elif caso == "c":
        t_c = np.arange(-6, 1 + delta, delta)
        t_ch = np.arange(-1, 4 + delta, delta)

        h_t = np.exp(t_c) * u(1 - t_c)
        x_t = u(t_ch + 1) - u(t_ch - 4)

        y_tpy = np.convolve(x_t, h_t) * delta
        len_yc = len(x_t) + len(h_t) - 1
        t_y = np.arange(t_ch[0] + t_c[0], t_ch[0] + t_c[0] + len_yc * delta, delta)

        t1_c = np.arange(-6, 0, delta)
        t2_c = np.arange(0, 5, delta)
        t3_c = np.arange(5, 10, delta)

        y1_c = np.exp(t1_c + 1) - np.exp(t1_c - 4)
        y2_c = np.exp(1) - np.exp(t2_c - 4)
        y3_c = np.zeros_like(t3_c)

        y_tm = np.concatenate((y1_c, y2_c, y3_c))
        t_m = np.concatenate((t1_c, t2_c, t3_c))

    # --- GRAFICAR ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Comparación de convoluciones — Caso {caso.upper()}", fontsize=16, fontweight="bold")

    # Señales originales
    axs[0,0].plot(t_x if caso=="a" else t_hb if caso=="b" else t_ch, x_t, label='x(t)')
    axs[0,0].plot(t_h if caso=="a" else t_b if caso=="b" else t_c, h_t, label='h(t)')
    axs[0,0].set_title("Señales originales")
    axs[0,0].legend(); axs[0,0].grid(True)

    # Convolución con Python
    axs[0,1].plot(t_y, y_tpy, 'r', label='np.convolve')
    axs[0,1].set_title("Convolución con np.convolve")
    axs[0,1].legend(); axs[0,1].grid(True)

    # Convolución manual
    axs[1,0].plot(t_m, y_tm, 'g', label='Manual')
    axs[1,0].set_title("Convolución manual")
    axs[1,0].legend(); axs[1,0].grid(True)

    # Comparación
    axs[1,1].plot(t_y, y_tpy, 'r', label='Python')
    axs[1,1].plot(t_m, y_tm, 'g--', label='Manual')
    axs[1,1].set_title("Comparación manual vs np.convolve")
    axs[1,1].legend(); axs[1,1].grid(True)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    visor.pyplot(fig, clear_figure=True)
