from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import whisper
import tempfile
import uvicorn
import os
import warnings

# Suppress FP16 CPU warning
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# Initialize app
app = FastAPI()

# Allow frontend to connect (CORS fix)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (safe for local dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper model
model = whisper.load_model("small")  # change to "base" or "tiny" if slower machine


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Receives an audio file from the frontend, transcribes it using Whisper,
    and returns the transcribed text as JSON.
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Transcribe with Whisper
        result = model.transcribe(tmp_path)
        text = result["text"]

        # Clean up
        os.remove(tmp_path)

        return JSONResponse({"text": text})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# Optional: serve your HTML frontend directly
@app.get("/")
async def frontend():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>🎙️ Live Speech to Text</title>
      <style>
        body {
          font-family: Arial, sans-serif;
          background: #121212;
          color: #fff;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          height: 100vh;
        }
        h1 { color: #00ffcc; }
        button {
          background: #00ffcc;
          color: #000;
          border: none;
          padding: 15px 30px;
          border-radius: 10px;
          cursor: pointer;
          font-size: 16px;
          margin-top: 20px;
        }
        button:disabled {
          background: #444;
          color: #aaa;
          cursor: not-allowed;
        }
        #transcript {
          margin-top: 30px;
          font-size: 1.4em;
          width: 80%;
          text-align: center;
          min-height: 100px;
          border: 1px solid #444;
          border-radius: 10px;
          padding: 15px;
          background: #1e1e1e;
        }
      </style>
    </head>
    <body>
      <h1>🎤 Live Speech to Text</h1>
      <button id="recordBtn">Start Recording</button>
      <div id="transcript">Your words will appear here...</div>

      <script>
        const recordBtn = document.getElementById("recordBtn");
        const transcriptDiv = document.getElementById("transcript");
        let mediaRecorder;
        let audioChunks = [];

        recordBtn.addEventListener("click", async () => {
          if (recordBtn.textContent === "Start Recording") {
            startRecording();
          } else {
            stopRecording();
          }
        });

        async function startRecording() {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          mediaRecorder = new MediaRecorder(stream);
          audioChunks = [];

          mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
          };

          mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
            const formData = new FormData();
            formData.append("file", audioBlob, "speech.wav");

            transcriptDiv.innerHTML = "⏳ Transcribing...";
            const response = await fetch("/transcribe", {
              method: "POST",
              body: formData
            });

            const data = await response.json();
            transcriptDiv.innerHTML = data.text || "❌ Error transcribing audio.";
          };

          mediaRecorder.start();
          recordBtn.textContent = "Stop Recording";
          transcriptDiv.innerHTML = "🎙️ Listening...";
        }

        function stopRecording() {
          mediaRecorder.stop();
          recordBtn.textContent = "Start Recording";
        }
      </script>
    </body>
    </html>
    """)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
