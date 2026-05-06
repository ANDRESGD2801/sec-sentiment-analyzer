"""
SEC Filing Sentiment Analyzer
Uses Qwen2-0.5B-Instruct locally via Hugging Face Transformers.
"""

import re
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
_tokenizer = None
_model     = None


def load_model():
    global _tokenizer, _model
    if _model is None:
        print(f"Loading {MODEL_NAME}…")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        _model.eval()
        print("Model ready.")
    return _tokenizer, _model


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------

def _read_pdf(path: str) -> str:
    import fitz
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)


def _read_text(path: str) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            with open(path, encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    return ""


def parse_file(path: str) -> str:
    return _read_pdf(path) if path.lower().endswith(".pdf") else _read_text(path)


# ---------------------------------------------------------------------------
# Section detection & extraction
# ---------------------------------------------------------------------------

SECTION_PATTERNS = {
    "Item 1 – Business":             r"item\s+1[\.\s]+business",
    "Item 1A – Risk Factors":        r"item\s+1a[\.\s]+risk\s+factor",
    "Item 2 – Properties":           r"item\s+2[\.\s]+properties",
    "Item 3 – Legal Proceedings":    r"item\s+3[\.\s]+legal\s+proceedings",
    "Item 7 – MD&A":                 r"item\s+7[\.\s]+management",
    "Item 7A – Market Risk":         r"item\s+7a[\.\s]+quantitative",
    "Item 8 – Financial Statements": r"item\s+8[\.\s]+financial\s+statements",
}


def detect_sections(text: str) -> list[str]:
    found = ["Full Document"]
    for name, pat in SECTION_PATTERNS.items():
        if re.search(pat, text, re.IGNORECASE):
            found.append(name)
    return found


def extract_section(text: str, section_name: str) -> str:
    if section_name == "Full Document" or section_name not in SECTION_PATTERNS:
        return text
    pat = SECTION_PATTERNS[section_name]
    m   = re.search(pat, text, re.IGNORECASE)
    if not m:
        return text
    start      = m.start()
    candidates = []
    for name, other_pat in SECTION_PATTERNS.items():
        if name == section_name:
            continue
        om = re.search(other_pat, text[start + 100:], re.IGNORECASE)
        if om:
            candidates.append(start + 100 + om.start())
    end = min(candidates) if candidates else len(text)
    return text[start:end].strip()


# ---------------------------------------------------------------------------
# Sentiment helpers
# ---------------------------------------------------------------------------

CHUNK_SIZE   = 1_200   # chars per chunk for logit method
MAX_CHUNKS   = 10      # max chunks to process (avoids very long waits)
MAX_LLM_CHARS = 2_500  # chars sent to instruct model


def _logit_scores_single(text: str) -> dict[str, float]:
    """Return softmax probabilities for Positive/Negative/Neutral on one chunk."""
    tok, mdl = load_model()
    prompt = (
        "Question: What is the overall sentiment of this SEC filing excerpt "
        "(positive, negative, or neutral)?\n"
        f"Excerpt: {text}\n"
        "Answer:"
    )
    input_ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        final_logits = mdl(input_ids).logits[0, -1]

    ids = {
        "Positive": tok(" positive")["input_ids"][0],
        "Negative": tok(" negative")["input_ids"][0],
        "Neutral":  tok(" neutral")["input_ids"][0],
    }
    raw   = torch.tensor([final_logits[v] for v in ids.values()])
    probs = raw.softmax(dim=0)
    return {label: float(p) for label, p in zip(ids.keys(), probs)}


def logit_sentiment(text: str, progress=gr.Progress()) -> tuple[str, dict[str, float]]:
    """
    Chunk the section, run logit comparison on each chunk,
    and return the averaged probabilities.
    """
    words  = text.split()
    chunks = []
    buf    = []
    length = 0
    for word in words:
        buf.append(word)
        length += len(word) + 1
        if length >= CHUNK_SIZE:
            chunks.append(" ".join(buf))
            buf, length = [], 0
    if buf:
        chunks.append(" ".join(buf))

    chunks = chunks[:MAX_CHUNKS]

    totals = {"Positive": 0.0, "Negative": 0.0, "Neutral": 0.0}
    for i, chunk in enumerate(chunks):
        progress((i + 1) / len(chunks), desc=f"Analyzing chunk {i+1}/{len(chunks)}")
        scores = _logit_scores_single(chunk)
        for k in totals:
            totals[k] += scores[k]

    n      = len(chunks)
    avg    = {k: v / n for k, v in totals.items()}
    best   = max(avg, key=avg.get)
    return best, avg


def instruct_sentiment(text: str) -> tuple[str, str]:
    """
    Ask the instruct model for a sentiment label + explanation.
    Returns (label, full_response).
    """
    tok, mdl = load_model()
    excerpt  = text[:MAX_LLM_CHARS].strip()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a financial analyst specializing in SEC filings. "
                "Read the provided excerpt and output:\n"
                "1. Sentiment: <Positive | Negative | Neutral>\n"
                "2. Reason: <one sentence>\n"
                "Use exactly that format."
            ),
        },
        {
            "role": "user",
            "content": f"SEC filing excerpt:\n\n{excerpt}",
        },
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

    # Extract the label from the structured response
    match = re.search(r"Sentiment:\s*(Positive|Negative|Neutral)", response, re.IGNORECASE)
    label = match.group(1).capitalize() if match else "Unknown"
    return label, response


# ---------------------------------------------------------------------------
# Gradio handlers
# ---------------------------------------------------------------------------

def on_upload(file):
    if file is None:
        return "", gr.update(choices=["Full Document"], value="Full Document"), "", ""
    # Gradio 5+ delivers a filepath string; older versions gave an object with .name
    path = file if isinstance(file, str) else file.name
    try:
        text = parse_file(path)
    except Exception as e:
        err = f"Error reading file: {e}"
        return "", gr.update(choices=["Full Document"], value="Full Document"), err, err

    if not text.strip():
        msg = "The file appears to be empty or could not be parsed."
        return "", gr.update(choices=["Full Document"], value="Full Document"), msg, msg

    sections = detect_sections(text)
    n_words  = len(text.split())
    n_chars  = len(text)
    stats    = f"Document loaded: **{n_words:,} words** / {n_chars:,} characters"
    preview  = text[:2_000] + ("\n\n[truncated…]" if n_chars > 2_000 else "")
    return text, gr.update(choices=sections, value=sections[0]), stats, preview


def on_analyze(full_text: str, section_name: str, progress=gr.Progress()):
    if not full_text.strip():
        return (
            "Upload a file first.",
            "",
            {},
            "",
        )

    section_text = extract_section(full_text, section_name)
    n_words      = len(section_text.split())
    preview      = section_text[:1_500] + ("\n\n[truncated…]" if len(section_text) > 1_500 else "")

    # ── Logit method ──
    progress(0, desc="Starting logit analysis…")
    logit_label, logit_scores = logit_sentiment(section_text, progress)

    # ── Instruct method ──
    progress(0.95, desc="Running instruct model…")
    inst_label, inst_response = instruct_sentiment(section_text)
    progress(1.0, desc="Done")

    # Stats line
    stats = (
        f"**Section:** {section_name} | "
        f"**Words analyzed:** {n_words:,} | "
        f"**Chunks processed:** {min(len(section_text.split()) // (CHUNK_SIZE // 5), MAX_CHUNKS)}"
    )

    # Agreement indicator
    agreement = "✅ Both methods agree" if logit_label == inst_label else "⚠️ Methods disagree"
    summary   = f"**Logit → {logit_label}** | **Instruct → {inst_label}** | {agreement}"

    return inst_response, summary, logit_scores, preview


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

CSS = """
.result-box { border-left: 4px solid #6366f1; padding: 12px; border-radius: 6px; }
"""

with gr.Blocks(title="SEC Sentiment Analyzer", theme=gr.themes.Soft(), css=CSS) as demo:

    gr.Markdown(
        "# SEC Filing Sentiment Analyzer\n"
        f"**Model:** `{MODEL_NAME}` — runs 100 % locally  |  "
        "Two analysis methods: logit-based (notebook approach) + instruct prompt"
    )

    _state = gr.State("")

    # ── Row 1: upload + controls ──────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload SEC Filing (PDF / TXT / HTM)",
                file_types=[".pdf", ".txt", ".htm", ".html"],
            )
            doc_stats  = gr.Markdown("")
            section_dd = gr.Dropdown(
                choices=["Full Document"],
                value="Full Document",
                label="Section to analyze",
            )
            analyze_btn = gr.Button("Analyze Sentiment", variant="primary", size="lg")
            gr.Markdown("_First run downloads ~1 GB of model weights._")

        with gr.Column(scale=2):
            preview_tb = gr.Textbox(
                label="Document / Section Preview",
                lines=12,
                interactive=False,
            )

    # ── Row 2: results ───────────────────────────────────────────────────
    gr.Markdown("## Results")

    summary_md = gr.Markdown("")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Instruct Model")
            inst_out = gr.Textbox(
                label="Full response",
                lines=5,
                interactive=False,
                elem_classes=["result-box"],
            )

        with gr.Column():
            gr.Markdown("### Logit-based Confidence")
            label_out = gr.Label(
                label="Sentiment probabilities",
                num_top_classes=3,
            )

    analysis_stats = gr.Markdown("")

    # ── Wiring ───────────────────────────────────────────────────────────
    file_input.change(
        fn=on_upload,
        inputs=[file_input],
        outputs=[_state, section_dd, doc_stats, preview_tb],
    )

    analyze_btn.click(
        fn=on_analyze,
        inputs=[_state, section_dd],
        outputs=[inst_out, summary_md, label_out, preview_tb],
    )


if __name__ == "__main__":
    load_model()
    demo.launch()
