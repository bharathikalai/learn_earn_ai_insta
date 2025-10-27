"""
# 🎙️ Live Speech-to-Text Web App (OpenAI Whisper + FastAPI)

This single-file project lets you record your voice from a web page
and converts it to text using OpenAI’s **Whisper** model (fully open source).

---

## 🧩 Features
✅ Record voice from your browser  
✅ Automatically transcribe speech to text  
✅ Powered by **FastAPI + Whisper**  
✅ Works offline after model download  

---

## ⚙️ Requirements

- Python 3.9 or newer  
- Microphone access in browser  
- (Optional) GPU for faster transcription

---

## 🧠 Installation

```bash
pip install fastapi uvicorn git+https://github.com/openai/whisper.git torch soundfile python-multipart

python3 -m pip install git+https://github.com/openai/whisper.git

