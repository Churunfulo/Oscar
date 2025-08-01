import gradio as gr
import cv2
import numpy as np
from PIL import Image

# Diccionario de equivalencias para onzas
ONZAS = {
    "Onza americana (56.7 g)": 56.7,
    "Onza española (60 g)": 60.0
}

# Función principal
def contar_garbanzos(imagen, tipo_onza):
    imagen_cv = cv2.cvtColor(np.array(imagen), cv2.COLOR_RGB2BGR)
    gris = cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gris, 100, 255, cv2.THRESH_BINARY_INV)
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_garbanzos = len(contornos)

    salida = imagen_cv.copy()
    cv2.drawContours(salida, contornos, -1, (0, 255, 0), 2)

    calibre = round(total_garbanzos / 2, 2)

    salida_rgb = cv2.cvtColor(salida, cv2.COLOR_BGR2RGB)
    salida_pil = Image.fromarray(salida_rgb)

    resultado_texto = (
        f"🔢 Garbanzos detectados: {total_garbanzos}\n"
        f"📏 Calibre estimado: {calibre} o.a ({tipo_onza})"
    )
    return salida_pil, resultado_texto

# Interfaz Gradio
with gr.Blocks(css="""
body {
    background-color: #F2F6FF;
    color: #1C2C48;
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3, p {
    color: #1C2C48;
}
.gr-button {
    background-color: #6985A6 !important;
    color: white !important;
    border-radius: 6px;
    border: none;
}
.gr-textbox, .gr-number, .gr-dropdown, .gr-image {
    border: 1px solid #292A2C;
    border-radius: 6px;
    background-color: white;
    color: #1C2C48;
}
""") as app:

    with gr.Row():
        gr.Markdown("""
        <img src="cargo control logotipo.png" alt="Logo Cargo Control" style="height:60px; width:auto;">
        """)
        gr.Markdown("<h2 style='text-align: right; margin-top: 20px;'>CARGO CONTROL</h2>")

    gr.Markdown("<h1 style='text-align: center;'>Digital System Chickpea Sizing Tool</h1>")
    gr.Markdown("""
    <p style='text-align: center;'>Sube una imagen de garbanzos sobre fondo neutro y selecciona el tipo de onza para calcular el calibre estimado.</p>
    """)

    with gr.Row():
        dropdown_onza = gr.Dropdown(
            choices=list(ONZAS.keys()),
            label="⚖️ Tipo de Onza",
            value="Onza americana (56.7 g)"
        )

    imagen_input = gr.Image(type="pil", label="📸 Imagen de muestra")
    imagen_salida = gr.Image(type="pil", label="🟢 Imagen procesada con contornos")
    resultado = gr.Textbox(label="📊 Resultados")

    gr.Button("Analizar").click(
        fn=contar_garbanzos,
        inputs=[imagen_input, dropdown_onza],
        outputs=[imagen_salida, resultado]
    )

    gr.Markdown("<p style='font-size:12px; text-align: center;'>Desarrollado por M.C. Oscar Soto Espinoza</p>")

app.launch()
