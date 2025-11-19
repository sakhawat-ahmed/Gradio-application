import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load NLLB model
model_name = "facebook/nllb-200-3.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Supported languages with NLLB language codes
LANGUAGES = {
    "Acehnese": "ace_Arab", "Afrikaans": "afr_Latn", "Akan": "aka_Latn", "Amharic": "amh_Ethi", "Armenian": "hye_Armn",
    "Assamese": "asm_Beng", "Asturian": "ast_Latn", "Awadhi": "awa_Deva", "Ayacucho Quechua": "quy_Latn", "Balinese": "ban_Latn",
    "Bambara": "bam_Latn", "Banjar": "bjn_Arab", "Bashkir": "bak_Cyrl", "Basque": "eus_Latn", "Belarusian": "bel_Cyrl",
    "Bemba": "bem_Latn", "Bengali": "ben_Beng", "Bhojpuri": "bho_Deva", "Bosnian": "bos_Latn", "Buginese": "bug_Latn",
    "Bulgarian": "bul_Cyrl", "Burmese": "mya_Mymr", "Catalan": "cat_Latn", "Cebuano": "ceb_Latn", "Central Atlas Tamazight": "tzm_Tfng",
    "Central Aymara": "ayr_Latn", "Central Kanuri": "knc_Latn", "Central Kurdish": "ckb_Arab", "Chhattisgarhi": "hne_Deva", "Chinese": "zho_Hans",
    "Chittagonian": "ctg_Beng", "Crimean Tatar": "crh_Latn", "Croatian": "hrv_Latn", "Czech": "ces_Latn", "Danish": "dan_Latn",
    "Dari": "prs_Arab", "Dutch": "nld_Latn", "Dyula": "dyu_Latn", "Dzongkha": "dzo_Tibt", "Eastern Panjabi": "pan_Guru",
    "Eastern Yiddish": "ydd_Hebr", "Egyptian Arabic": "arz_Arab", "English": "eng_Latn", "Esperanto": "epo_Latn", "Estonian": "est_Latn",
    "Ewe": "ewe_Latn", "Faroese": "fao_Latn", "Fijian": "fij_Latn", "Finnish": "fin_Latn", "Fon": "fon_Latn",
    "French": "fra_Latn", "Friulian": "fur_Latn", "Galician": "glg_Latn", "Ganda": "lug_Latn", "Georgian": "kat_Geor",
    "German": "deu_Latn", "Greek": "ell_Grek", "Guarani": "grn_Latn", "Gujarati": "guj_Gujr", "Haitian Creole": "hat_Latn",
    "Halh Mongolian": "khk_Cyrl", "Hausa": "hau_Latn", "Hebrew": "heb_Hebr", "Hindi": "hin_Deva", "Hmong Njua": "hnj_Latn",
    "Hungarian": "hun_Latn", "Icelandic": "isl_Latn", "Igbo": "ibo_Latn", "Ilocano": "ilo_Latn", "Indonesian": "ind_Latn",
    "Irish": "gle_Latn", "Italian": "ita_Latn", "Japanese": "jpn_Jpan", "Javanese": "jav_Latn", "Jingpho": "kac_Latn",
    "Kamba": "kam_Latn", "Kannada": "kan_Knda", "Kashmiri": "kas_Arab", "Kazakh": "kaz_Cyrl", "Khmer": "khm_Khmr",
    "Kikongo": "kon_Latn", "Kikuyu": "kik_Latn", "Kimbundu": "kmb_Latn", "Kinyarwanda": "kin_Latn", "Kirghiz": "kir_Cyrl",
    "Kongo": "kon_Latn", "Korean": "kor_Hang", "Kurdish": "kur_Latn", "Lao": "lao_Laoo", "Latgalian": "ltg_Latn",
    "Latin": "lat_Latn", "Latvian": "lav_Latn", "Ligurian": "lij_Latn", "Limburgish": "lim_Latn", "Lingala": "lin_Latn",
    "Lithuanian": "lit_Latn", "Lombard": "lmo_Latn", "Luba-Kasai": "lua_Latn", "Luo": "luo_Latn", "Luxembourgish": "ltz_Latn",
    "Macedonian": "mkd_Cyrl", "Magahi": "mag_Deva", "Maithili": "mai_Deva", "Malagasy": "plt_Latn", "Malay": "msa_Latn",
    "Malayalam": "mal_Mlym", "Maltese": "mlt_Latn", "Manipuri": "mni_Beng", "Marathi": "mar_Deva", "Minangkabau": "min_Arab",
    "Mizo": "lus_Latn", "Modern Greek": "ell_Grek", "Modern Standard Arabic": "arb_Arab", "Moroccan Arabic": "ary_Arab", "Mossi": "mos_Latn",
    "Nepali": "npi_Deva", "Nigerian Fulfulde": "fuv_Latn", "North Azerbaijani": "azj_Latn", "North Levantine Arabic": "apc_Arab", "Northern Kurdish": "kmr_Latn",
    "Northern Sotho": "nso_Latn", "Northern Uzbek": "uzn_Latn", "Norwegian": "nob_Latn", "Nyanja": "nya_Latn", "Occitan": "oci_Latn",
    "Odia": "ory_Orya", "Pangasinan": "pag_Latn", "Papiamento": "pap_Latn", "Pashto": "pbt_Arab", "Persian": "pes_Arab",
    "Polish": "pol_Latn", "Portuguese": "por_Latn", "Romanian": "ron_Latn", "Rundi": "run_Latn", "Russian": "rus_Cyrl",
    "Saraiki": "skr_Arab", "Sardinian": "srd_Latn", "Scottish Gaelic": "gla_Latn", "Serbian": "srp_Cyrl", "Shan": "shn_Mymr",
    "Shona": "sna_Latn", "Sicilian": "scn_Latn", "Silesian": "szl_Latn", "Sindhi": "snd_Arab", "Sinhala": "sin_Sinh",
    "Slovak": "slk_Latn", "Slovenian": "slv_Latn", "Somali": "som_Latn", "South Azerbaijani": "azb_Arab", "Southern Pashto": "pbt_Arab",
    "Southern Sotho": "sot_Latn", "Spanish": "spa_Latn", "Standard Latvian": "lvs_Latn", "Standard Malay": "zsm_Latn", "Sundanese": "sun_Latn",
    "Swahili": "swh_Latn", "Swati": "ssw_Latn", "Swedish": "swe_Latn", "Tagalog": "tgl_Latn", "Tajik": "tgk_Cyrl",
    "Tamasheq": "taq_Latn", "Tamil": "tam_Taml", "Tatar": "tat_Cyrl", "Telugu": "tel_Telu", "Thai": "tha_Thai",
    "Tibetan": "bod_Tibt", "Tigrinya": "tir_Ethi", "Tok Pisin": "tpi_Latn", "Tosk Albanian": "als_Latn", "Tsonga": "tso_Latn",
    "Tswana": "tsn_Latn", "Tumbuka": "tum_Latn", "Turkish": "tur_Latn", "Turkmen": "tuk_Latn", "Uighur": "uig_Arab",
    "Ukrainian": "ukr_Cyrl", "Umbundu": "umb_Latn", "Urdu": "urd_Arab", "Venetian": "vec_Latn", "Vietnamese": "vie_Latn",
    "Waray": "war_Latn", "Welsh": "cym_Latn", "West Central Oromo": "gaz_Latn", "Western Persian": "pes_Arab", "Wolof": "wol_Latn",
    "Xhosa": "xho_Latn", "Yoruba": "yor_Latn", "Zulu": "zul_Latn"
}

def translate(text, src, tgt):
    if not text:
        return ""

    tgt_code = LANGUAGES[tgt]
    inputs = tokenizer(text, return_tensors="pt")
    forced_id = tokenizer.convert_tokens_to_ids(tgt_code)
    generated = model.generate(**inputs, forced_bos_token_id=forced_id)
    return tokenizer.decode(generated[0], skip_special_tokens=True)

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

            # Input and output remain the same as current design
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