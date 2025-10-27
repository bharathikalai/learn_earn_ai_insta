"""
# 🎙️ Live Speech-to-Text Web App (OpenAI Whisper + FastAPI)  speech-to-text AI (Whisper)

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
# 1. Go to your project folder
cd ~/learn-ai/learn_earn_ai_insta/wishper

# 2. Create a clean virtual environment
python3 -m venv venv

# 3. Activate the virtual environment
source venv/bin/activate

# 4. Upgrade pip and build tools
pip install --upgrade pip setuptools wheel

# 5. Install Whisper + all dependencies
pip install git+https://github.com/openai/whisper.git torch fastapi uvicorn soundfile python-multipart

# 6. Verify installation
python3 - <<'EOF'
import whisper
print("✅ Whisper installed and working!")
print("Available:", dir(whisper)[:8])
EOF



# open source code
```
https://github.com/openai/whisper
```