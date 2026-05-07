"""
Analizador de Sentimiento — SEC Filings
Modelo: Qwen2-0.5B-Instruct (local, sin internet en inferencia)

Dos métodos:
  1. Logit: compara probabilidades del siguiente token (Positive/Negative/Neutral)
  2. Instruct: pide al modelo una respuesta estructurada en lenguaje natural
"""

import re
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Carga del modelo (lazy — solo la primera vez que se necesita)
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
_tokenizer = None
_model = None
_label_ids = None  # IDs de token para Positive / Negative / Neutral


def load_model():
    global _tokenizer, _model, _label_ids
    if _model is None:
        print(f"Cargando {MODEL_NAME} — primera ejecución descarga ~1 GB...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        _model.eval()
        _label_ids = _compute_label_ids(_tokenizer)
        print(f"Modelo listo.  Label token IDs: {_label_ids}")
    return _tokenizer, _model, _label_ids


def _compute_label_ids(tok):
    """
    Obtiene el ID del primer sub-token de cada etiqueta de sentimiento.
    Con prefijo de espacio (' Positive') para respetar el convenio BPE.
    """
    ids = {}
    for label in ("Positive", "Negative", "Neutral"):
        encoded = tok(" " + label, add_special_tokens=False)["input_ids"]
        ids[label] = encoded[0]  # primer sub-token es suficientemente distintivo
    return ids


# ---------------------------------------------------------------------------
# Lectura de archivos
# ---------------------------------------------------------------------------

def read_file(path: str) -> str:
    if path.lower().endswith(".pdf"):
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(path)
            return "\n".join(p.get_text() for p in doc)
        except ImportError:
            return "ERROR: Instala PyMuPDF para leer PDFs:  pip install pymupdf"
        except Exception as e:
            return f"ERROR al abrir PDF: {e}"

    for enc in ("utf-8", "latin-1"):
        try:
            with open(path, encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    return ""


# ---------------------------------------------------------------------------
# Detección y extracción de secciones de un 10-K
# ---------------------------------------------------------------------------

SECTIONS = {
    "Item 1 – Business":             r"item\s+1[\.\s]+business",
    "Item 1A – Risk Factors":        r"item\s+1a[\.\s]+risk\s+factor",
    "Item 2 – Properties":           r"item\s+2[\.\s]+properties",
    "Item 3 – Legal Proceedings":    r"item\s+3[\.\s]+legal\s+proceedings",
    "Item 7 – MD&A":                 r"item\s+7[\.\s]+management",
    "Item 7A – Market Risk":         r"item\s+7a[\.\s]+quantitative",
    "Item 8 – Financial Statements": r"item\s+8[\.\s]+financial",
}


def detect_sections(text: str) -> list:
    found = ["Full Document"]
    for name, pat in SECTIONS.items():
        if re.search(pat, text, re.IGNORECASE):
            found.append(name)
    return found


def extract_section(text: str, name: str) -> str:
    """Extrae el texto de una sección hasta que empieza la siguiente."""
    if name == "Full Document" or name not in SECTIONS:
        return text
    pat = SECTIONS[name]
    m = re.search(pat, text, re.IGNORECASE)
    if not m:
        return text
    start = m.start()
    # Busca dónde empieza la siguiente sección detectada
    ends = []
    for other_name, other_pat in SECTIONS.items():
        if other_name == name:
            continue
        om = re.search(other_pat, text[start + 200:], re.IGNORECASE)
        if om:
            ends.append(start + 200 + om.start())
    return text[start: min(ends) if ends else len(text)].strip()


# ---------------------------------------------------------------------------
# Método 1: Logit comparison (del notebook del profe)
# ---------------------------------------------------------------------------

CHUNK_CHARS = 1_200   # caracteres por chunk
MAX_CHUNKS  = 10      # máximo de chunks para no tardar demasiado


def logit_sentiment(text: str) -> tuple:
    """
    Divide el texto en chunks, compara los logits del último token para
    Positive/Negative/Neutral y promedia las probabilidades.
    Retorna: (etiqueta_ganadora, {etiqueta: probabilidad})
    """
    tok, mdl, label_ids = load_model()

    # Dividir en chunks por palabras
    chunks, buf, n = [], [], 0
    for word in text.split():
        buf.append(word)
        n += len(word) + 1
        if n >= CHUNK_CHARS:
            chunks.append(" ".join(buf))
            buf, n = [], 0
    if buf:
        chunks.append(" ".join(buf))
    chunks = chunks[:MAX_CHUNKS]

    totals = {k: 0.0 for k in label_ids}

    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}/{len(chunks)}...")
        prompt = (
            "Question: What is the overall sentiment of this SEC filing excerpt "
            "(Positive, Negative, or Neutral)?\n"
            f"Excerpt: {chunk}\n"
            "Answer:"
        )
        input_ids = tok(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            last_logits = mdl(input_ids).logits[0, -1]

        # Softmax solo sobre los 3 tokens de interés
        raw = torch.tensor([last_logits[v].item() for v in label_ids.values()])
        probs = raw.softmax(dim=0)
        for label, p in zip(label_ids.keys(), probs):
            totals[label] += float(p)

    n = len(chunks)
    avg = {k: v / n for k, v in totals.items()}
    best = max(avg, key=avg.get)
    return best, avg


# ---------------------------------------------------------------------------
# Método 2: Instruct (chat template)
# ---------------------------------------------------------------------------

MAX_INSTRUCT_CHARS = 2_500  # recorta para no exceder el contexto del modelo pequeño


def instruct_sentiment(text: str) -> tuple:
    """
    Usa el chat template de Qwen2-Instruct para pedir un análisis estructurado.
    Retorna: (etiqueta, respuesta_completa)
    """
    tok, mdl, _ = load_model()
    excerpt = text[:MAX_INSTRUCT_CHARS].strip()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a financial analyst specializing in SEC filings. "
                "Analyze the excerpt and reply using EXACTLY this format:\n"
                "Sentiment: <Positive | Negative | Neutral>\n"
                "Reason: <one sentence explaining the sentiment>"
            ),
        },
        {"role": "user", "content": f"SEC filing excerpt:\n\n{excerpt}"},
    ]

    prompt_str = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt_str, return_tensors="pt")

    with torch.no_grad():
        out_ids = mdl.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )

    new_tokens = out_ids[0][inputs["input_ids"].shape[1]:]
    response = tok.decode(new_tokens, skip_special_tokens=True).strip()

    m = re.search(r"Sentiment:\s*(Positive|Negative|Neutral)", response, re.IGNORECASE)
    label = m.group(1).capitalize() if m else "Unknown"
    return label, response


# ---------------------------------------------------------------------------
# Manejadores de Gradio
# ---------------------------------------------------------------------------

def on_upload(file):
    """Se ejecuta cuando el usuario sube un archivo."""
    if file is None:
        return "", gr.update(choices=["Full Document"], value="Full Document"), "Sube un archivo para comenzar."

    path = file if isinstance(file, str) else file.name
    text = read_file(path)

    if not text.strip():
        return "", gr.update(choices=["Full Document"], value="Full Document"), "El archivo está vacío o no se pudo leer."

    sections = detect_sections(text)
    preview = text[:3_000] + ("\n\n[... documento truncado para vista previa ...]" if len(text) > 3_000 else "")
    info = f"Documento cargado: **{len(text.split()):,} palabras** | Secciones detectadas: {len(sections)-1}"
    return text, gr.update(choices=sections, value=sections[0]), info + "\n\n" + preview


def on_analyze(full_text: str, section_name: str):
    """Se ejecuta al presionar 'Analizar Sentimiento'."""
    if not full_text.strip():
        return "Primero sube un archivo SEC.", "", {}, ""

    section_text = extract_section(full_text, section_name)
    word_count = len(section_text.split())
    section_preview = (
        section_text[:2_000] + "\n\n[... sección truncada para vista previa ...]"
        if len(section_text) > 2_000 else section_text
    )

    print(f"\nAnalizando sección '{section_name}' ({word_count} palabras)...")

    # Método logit
    logit_label, logit_scores = logit_sentiment(section_text)

    # Método instruct
    inst_label, inst_response = instruct_sentiment(section_text)

    # Resumen
    agree = "✅ Ambos métodos coinciden" if logit_label == inst_label else "⚠️ Los métodos difieren"
    summary = (
        f"**Logit → {logit_label}** | **Instruct → {inst_label}** | {agree}  \n"
        f"Sección: *{section_name}* · {word_count:,} palabras"
    )

    return inst_response, summary, logit_scores, section_preview


# ---------------------------------------------------------------------------
# Interfaz Gradio
# ---------------------------------------------------------------------------

with gr.Blocks(title="SEC Sentiment Analyzer", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        "# Analizador de Sentimiento — Archivos SEC\n"
        f"**Modelo:** `{MODEL_NAME}` &nbsp;·&nbsp; Corre 100 % en local\n\n"
        "Sube un archivo de texto del SEC (10-K, 10-Q en TXT, HTM o PDF) y selecciona "
        "una sección para analizar su sentimiento con dos métodos diferentes."
    )

    _state = gr.State("")  # guarda el texto completo del documento

    # ── Fila 1: carga de archivo y controles ─────────────────────────────
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Archivo SEC (PDF / TXT / HTM)",
                file_types=[".pdf", ".txt", ".htm", ".html"],
            )
            section_dd = gr.Dropdown(
                choices=["Full Document"],
                value="Full Document",
                label="Sección a analizar",
            )
            analyze_btn = gr.Button("Analizar Sentimiento", variant="primary", size="lg")
            gr.Markdown("*Primera ejecución: descarga ~1 GB del modelo (solo una vez).*")

        with gr.Column(scale=2):
            preview_box = gr.Textbox(
                label="Vista previa",
                lines=16,
                interactive=False,
            )

    # ── Fila 2: resultados ───────────────────────────────────────────────
    gr.Markdown("---")
    summary_md = gr.Markdown("")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Método Instruct")
            inst_out = gr.Textbox(
                label="Respuesta del modelo",
                lines=6,
                interactive=False,
            )
        with gr.Column():
            gr.Markdown("### Método Logit (probabilidades)")
            label_out = gr.Label(
                label="Confianza por sentimiento",
                num_top_classes=3,
            )

    # ── Conexiones ───────────────────────────────────────────────────────
    file_input.change(
        fn=on_upload,
        inputs=[file_input],
        outputs=[_state, section_dd, preview_box],
    )

    analyze_btn.click(
        fn=on_analyze,
        inputs=[_state, section_dd],
        outputs=[inst_out, summary_md, label_out, preview_box],
    )


if __name__ == "__main__":
    load_model()          # pre-carga el modelo al arrancar
    demo.launch()
