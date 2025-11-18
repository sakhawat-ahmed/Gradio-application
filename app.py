import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load NLLB model
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Supported languages with NLLB language codes
LANGUAGES = {
    "English": "eng_Latn",
    "Bengali": "ben_Beng",
    "Chinese": "zho_Hans",
    "Hindi": "hin_Deva",
    "Urdu": "urd_Arab",
    "Spanish": "spa_Latn",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Italian": "ita_Latn",
    "Portuguese": "por_Latn",
    "Japanese": "jpn_Jpan",
    "Korean": "kor_Hang",
    "Russian": "rus_Cyrl",
    "Arabic": "arb_Arab",
    "Persian": "pes_Arab",
    "Turkish": "tur_Latn",
    "Vietnamese": "vie_Latn",
    "Thai": "tha_Thai",
    "Dutch": "nld_Latn",
    "Swedish": "swe_Latn",
}

def translate(text, src, tgt):
    if not text:
        return ""

    tgt_code = LANGUAGES[tgt]

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt")

    # Correct forced BOS token id for NLLB
    forced_id = tokenizer.convert_tokens_to_ids(tgt_code)

    # Generate translation
    generated = model.generate(
        **inputs,
        forced_bos_token_id=forced_id
    )

    # Decode text
    return tokenizer.decode(generated[0], skip_special_tokens=True)

# ------------------
#  UI DESIGN ADDED
# ------------------

custom_css = """
    body {
        background: #ffffff;
        font-family: Arial, sans-serif;
    }

    .main-container {
        max-width: 900px;
        margin: auto;
        padding: 20px;
    }

    .title {
        font-size: 2rem;
        font-weight: 600;
        text-align: center;
        color: #202124;
    }

    .gt-box {
        border: 1px solid #dadce0;
        border-radius: 12px;
        padding: 0;
        display: flex;
        flex-direction: column;
        background: #fff;
    }

    .gt-row {
        display: flex;
        gap: 20px;
        margin-top: 20px;
    }

    .lang-select select {
        border: none !important;
        background: none !important;
        padding: 12px !important;
        font-size: 1rem !important;
        color: #202124 !important;
    }

    textarea {
        border: none !important;
        padding: 20px !important;
        resize: none !important;
        font-size: 1.1rem !important;
    }

    .swap-btn button {
        width: 42px !important;
        height: 42px !important;
        border-radius: 50% !important;
        border: 1px solid #dcdcdc !important;
        background: #ffffff !important;
        font-size: 1.2rem !important;
        transition: 0.2s ease-in-out;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }

    .swap-btn button:hover {
        background: #f1f3f4 !important;
        transform: rotate(180deg);
    }

    .translate-btn button {
        background: #1a73e8 !important;
        color: white !important;
        font-size: 1rem !important;
        padding: 12px 18px !important;
        border-radius: 8px !important;
    }
"""

# Gradio UI
def build_ui():
    with gr.Blocks(css=custom_css) as demo:
        gr.Markdown("<div class='title'>AI Translator</div>")

        with gr.Column(elem_classes="main-container"):

            with gr.Row(elem_classes="gt-row"):
                with gr.Column(scale=1):
                    src = gr.Dropdown(list(LANGUAGES.keys()), value="English", label="", elem_classes="lang-select")
                with gr.Column(scale=0):
                    swap_btn = gr.Button("⟲", elem_classes="swap-btn")
                with gr.Column(scale=1):
                    tgt = gr.Dropdown(list(LANGUAGES.keys()), value="Bengali", label="", elem_classes="lang-select")

            with gr.Row():
                input_text = gr.Textbox(lines=8, label="", placeholder="Enter text", elem_classes="gt-box")
                output_text = gr.Textbox(lines=8, label="", placeholder="Translation", elem_classes="gt-box")

            translate_btn = gr.Button("Translate", elem_classes="translate-btn")

            translate_btn.click(translate, inputs=[input_text, src, tgt], outputs=output_text)

            def swap_languages(a, b):
                return b, a

            swap_btn.click(swap_languages, inputs=[src, tgt], outputs=[src, tgt])

    return demo


demo = build_ui()

demo.launch()
