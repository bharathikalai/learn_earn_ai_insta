from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  # <-- ADD THIS
import numpy as np
import joblib
import librosa
import tempfile
import os

# ------------ CONFIG ------------
MODEL_PATH = "/home/raise/Documents/learn-ai/learn_earn_ai_insta/cricket-spike/cricket_impact_model.joblib"
LABELS = ["bat", "pad"]

# Load model once at startup
model = joblib.load(MODEL_PATH)

app = Flask(__name__, static_folder="static", static_url_path="")

# Enable CORS for all routes  <-- ADD THIS
CORS(app)


# ------------ FEATURE EXTRACTION ------------
def extract_features_from_filestorage(file_storage, n_mfcc=40):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        file_storage.save(tmp.name)
        y, sr = librosa.load(tmp.name, sr=None)
        y, _ = librosa.effects.trim(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        return mfcc_mean


# ------------ ROUTES ------------
@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded (field name 'audio' missing)."}), 400

    file = request.files["audio"]

    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        features = extract_features_from_filestorage(file)
        features = features.reshape(1, -1)

        probs = model.predict_proba(features)[0]
        probs = probs.tolist()

        pred_idx = int(np.argmax(probs))
        pred_label = LABELS[pred_idx]

        prob_dict = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

        return jsonify({
            "label": pred_label,
            "probabilities": prob_dict
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": "Error processing audio."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5008, debug=True)
