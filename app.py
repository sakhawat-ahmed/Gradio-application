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


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🌍 Universal Translator (NLLB-200)\nSupports ALL language pairs.")

    with gr.Row():
        src = gr.Dropdown(list(LANGUAGES.keys()), label="Source Language", value="English")
        tgt = gr.Dropdown(list(LANGUAGES.keys()), label="Target Language", value="Bengali")

    input_text = gr.Textbox(label="Enter text")
    output_text = gr.Textbox(label="Translation")

    translate_btn = gr.Button("Translate")
    translate_btn.click(translate, inputs=[input_text, src, tgt], outputs=output_text)

demo.launch()
