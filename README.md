# AI Translator Gradio App


## Overview
**AI Translator** is a web-based application built using **Python** and **Gradio** that provides **machine translation** functionality.  
The application uses the **facebook/nllb-200-3.3B** pre-trained model from **Hugging Face Transformers**, which supports translation between over 200 languages.  

The app allows users to input text, select source and target languages, and get accurate translations interactively.  

**Purpose:**  
This project demonstrates the integration of a **pre-trained NLP model** into an **interactive web interface**, fulfilling the requirements of a Gradio-based NLP assignment.

---

## Features
- **Text Translation:** Translate between 200+ languages.
- **Language Swap:** Easily swap source and target languages with one click.
- **Interactive Interface:** Instant display of translations.
- **User-Friendly UI:** Clean design with input/output text boxes and dropdowns.
- **Custom Styling:** Modern look using custom CSS.

---
**1. Application Structure**
Gradio-application/
│
├─ app.py # Main Gradio app
├─ requirements.txt # Dependencies
└─ README.md # Documentation


**2. Key Components**
- `translate(text, src, tgt)` function:  
  Handles translation using NLLB-200 model. Converts language names to Hugging Face codes and generates the translated text.
- Gradio UI:
  - Input textbox for the text to translate.
  - Dropdown menus for source and target languages.
  - Translate button to perform translation.
  - Swap button to swap languages quickly.
- Custom CSS:
  - Styles for input/output boxes, buttons, and overall layout.

---

## Supported Languages
The app supports **over 200 languages**. Some examples include:

- English, Bengali, Hindi, Spanish, French, Chinese, Arabic, Russian, Japanese, German, Portuguese, Italian, Korean, Urdu, Tamil, Telugu, Malayalam, etc.

> Full list is available in the `LANGUAGES` dictionary in `app.py`.

---

## Setup Instructions

1. **Clone the repository**
```bash
git clone <repository_url>
cd Gradio-application

*Create a virtual environment
python -m venv venv


*Activate the virtual environment
Linux/Mac:
source venv/bin/activate


Windows:
venv\Scripts\activate


*Install dependencies
pip install -r requirements.txt


*Run the application
python app.py
