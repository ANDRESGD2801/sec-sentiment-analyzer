"""
Analizador de Sentimiento — SEC Filings
Modelo: Qwen2-0.5B-Instruct (local, sin internet en inferencia)

Cómo correr:
    python app.py
    Luego abre: http://127.0.0.1:7860

El servidor sirve index.html en / y la API de Gradio en /gradio.
"""

import re
import os
import torch
import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Modelo (lazy — descarga ~1 GB la primera vez)
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
_tokenizer = None
_model     = None
_label_ids = None


def load_model():
    global _tokenizer, _model, _label_ids
    if _model is None:
        print(f"Cargando {MODEL_NAME}...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        _model.eval()
        _label_ids = _get_label_ids(_tokenizer)
        print(f"Modelo listo. Token IDs: {_label_ids}")
    return _tokenizer, _model, _label_ids


def _get_label_ids(tok):
    ids = {}
    for label in ("Positive", "Negative", "Neutral"):
        encoded = tok(" " + label, add_special_tokens=False)["input_ids"]
        ids[label] = encoded[0]
    return ids


# ---------------------------------------------------------------------------
# Lectura de archivos
# ---------------------------------------------------------------------------

def read_file(path: str) -> str:
    if path.lower().endswith(".pdf"):
        try:
            import fitz
            return "\n".join(p.get_text() for p in fitz.open(path))
        except ImportError:
            return "ERROR: instala pymupdf para leer PDFs"
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
# Secciones
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
    if name == "Full Document" or name not in SECTIONS:
        return text
    m = re.search(SECTIONS[name], text, re.IGNORECASE)
    if not m:
        return text
    start = m.start()
    ends  = []
    for other, pat in SECTIONS.items():
        if other == name:
            continue
        om = re.search(pat, text[start + 200:], re.IGNORECASE)
        if om:
            ends.append(start + 200 + om.start())
    return text[start: min(ends) if ends else len(text)].strip()


# ---------------------------------------------------------------------------
# Sentimiento: método logit
# ---------------------------------------------------------------------------

CHUNK_CHARS    = 1_200
MAX_CHUNKS     = 10


def logit_sentiment(text: str) -> tuple:
    tok, mdl, label_ids = load_model()

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
        print(f"  Logit chunk {i+1}/{len(chunks)}...")
        prompt = (
            "Question: What is the overall sentiment of this SEC filing excerpt "
            "(Positive, Negative, or Neutral)?\n"
            f"Excerpt: {chunk}\nAnswer:"
        )
        input_ids = tok(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            last_logits = mdl(input_ids).logits[0, -1]
        raw   = torch.tensor([last_logits[v].item() for v in label_ids.values()])
        probs = raw.softmax(dim=0)
        for label, p in zip(label_ids.keys(), probs):
            totals[label] += float(p)

    n   = len(chunks)
    avg = {k: v / n for k, v in totals.items()}
    return max(avg, key=avg.get), avg


# ---------------------------------------------------------------------------
# Sentimiento: método instruct
# ---------------------------------------------------------------------------

MAX_INSTRUCT_CHARS = 2_500


def instruct_sentiment(text: str) -> tuple:
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
    inputs     = tok(prompt_str, return_tensors="pt")

    with torch.no_grad():
        out_ids = mdl.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )

    new_tokens = out_ids[0][inputs["input_ids"].shape[1]:]
    response   = tok.decode(new_tokens, skip_special_tokens=True).strip()
    m          = re.search(r"Sentiment:\s*(Positive|Negative|Neutral)", response, re.IGNORECASE)
    label      = m.group(1).capitalize() if m else "Unknown"
    return label, response


# ---------------------------------------------------------------------------
# Handlers de Gradio (usados por la UI Gradio en /gradio y por el HTML)
# ---------------------------------------------------------------------------

def on_upload(file):
    if file is None:
        return "", gr.update(choices=["Full Document"], value="Full Document"), "Sube un archivo."
    path = file if isinstance(file, str) else file.name
    text = read_file(path)
    if not text.strip():
        return "", gr.update(choices=["Full Document"], value="Full Document"), "Archivo vacío."
    sections = detect_sections(text)
    preview  = text[:3_000] + ("\n\n[truncado...]" if len(text) > 3_000 else "")
    return text, gr.update(choices=sections, value=sections[0]), preview


def on_analyze(full_text: str, section_name: str):
    if not full_text.strip():
        return "Sube un archivo primero.", "", {}, ""

    section_text = extract_section(full_text, section_name)
    words        = len(section_text.split())
    preview      = section_text[:2_000] + ("\n\n[truncado...]" if len(section_text) > 2_000 else "")

    print(f"\nAnalizando '{section_name}' ({words:,} palabras)...")

    logit_label, logit_scores = logit_sentiment(section_text)
    inst_label,  inst_resp    = instruct_sentiment(section_text)

    agree   = "Ambos metodos coinciden" if logit_label == inst_label else "Los metodos difieren"
    summary = (
        f"**Logit: {logit_label}** | **Instruct: {inst_label}** | {agree}  \n"
        f"Seccion: *{section_name}* · {words:,} palabras"
    )
    return inst_resp, summary, logit_scores, preview


# ---------------------------------------------------------------------------
# Interfaz Gradio (accesible en /gradio para uso directo)
# ---------------------------------------------------------------------------

with gr.Blocks(title="SEC Sentiment Analyzer", theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# SEC Sentiment Analyzer\n**Modelo:** `{MODEL_NAME}`")
    _state = gr.State("")

    with gr.Row():
        with gr.Column(scale=1):
            file_in   = gr.File(label="Archivo SEC", file_types=[".pdf", ".txt", ".htm", ".html"])
            sec_dd    = gr.Dropdown(choices=["Full Document"], value="Full Document", label="Sección")
            anl_btn   = gr.Button("Analizar", variant="primary")
        with gr.Column(scale=2):
            prev_box  = gr.Textbox(label="Vista previa", lines=14, interactive=False)

    gr.Markdown("---")
    summ_md  = gr.Markdown("")
    with gr.Row():
        with gr.Column():
            inst_out = gr.Textbox(label="Respuesta Instruct", lines=5, interactive=False)
        with gr.Column():
            lbl_out  = gr.Label(label="Probabilidades Logit", num_top_classes=3)

    file_in.change(fn=on_upload, inputs=[file_in],       outputs=[_state, sec_dd, prev_box])
    anl_btn.click( fn=on_analyze, inputs=[_state, sec_dd], outputs=[inst_out, summ_md, lbl_out, prev_box])


# ---------------------------------------------------------------------------
# FastAPI: sirve index.html en / y Gradio en /gradio
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))

fapp = FastAPI()

@fapp.get("/")
def serve_index():
    return FileResponse(os.path.join(_HERE, "index.html"), media_type="text/html")

demo.queue()
gr.mount_gradio_app(fapp, demo, path="/gradio")


if __name__ == "__main__":
    load_model()
    print("\n" + "="*50)
    print("  Abre en tu navegador: http://127.0.0.1:7860")
    print("  UI Gradio alternativa: http://127.0.0.1:7860/gradio")
    print("="*50 + "\n")
    uvicorn.run(fapp, host="127.0.0.1", port=7860)
