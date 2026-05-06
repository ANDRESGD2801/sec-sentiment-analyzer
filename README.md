# SEC Filing Sentiment Analyzer

Analiza el sentimiento de archivos de la SEC usando **Qwen2-0.5B-Instruct** corriendo 100 % en local, con una interfaz web sencilla hecha en Gradio.

## Características

- Sube archivos PDF, TXT o HTML de la SEC (10-K, 10-Q, etc.).
- Selecciona una sección específica (Item 1, Risk Factors, MD&A, etc.) o analiza el documento completo.
- **Dos métodos de análisis**:
  1. **Instruct model** — usa el chat template de Qwen2-Instruct para obtener una respuesta en lenguaje natural.
  2. **Logit-based** — comparación directa de logits para Positive / Negative / Neutral (enfoque del notebook de referencia).

## Requisitos

- Python 3.10+
- ~1 GB de espacio en disco para los pesos del modelo (se descargan automáticamente en la primera ejecución desde Hugging Face).

## Instalación

```bash
# Clona el repo
git clone <tu-repo>
cd sec_sentiment

# Crea entorno virtual (recomendado)
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Instala dependencias
pip install -r requirements.txt
```

## Uso

```bash
python app.py
```

Abre el navegador en `http://127.0.0.1:7860`.

1. Sube un archivo de la SEC.
2. Elige la sección que quieres analizar en el menú desplegable.
3. Haz clic en **Analyze Sentiment**.

## Estructura

```
sec_sentiment/
├── app.py           # App principal (Gradio + Transformers)
├── requirements.txt
└── README.md
```

## Referencia

Basado en el notebook [text_models.ipynb](https://github.com/LuisLozanoM/AI_impacto_empresarial/blob/main/text_models.ipynb) del curso AI Impacto Empresarial.
