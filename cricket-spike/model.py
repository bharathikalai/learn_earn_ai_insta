import os
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "/home/raise/Documents/learn-ai/learn_earn_ai_insta/cricket-spike/data"
MODEL_PATH = "cricket_impact_model.joblib"

LABELS = {
    "bat": 0,
    "pad": 1
}


# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_features(file_path, n_mfcc=40):
    try:
        y, sr = librosa.load(file_path, sr=None)
        y, _ = librosa.effects.trim(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        return mfcc_mean
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# -----------------------------
# LOAD DATASET
# -----------------------------
def load_dataset():
    X, y = [], []

    for label_name, label_idx in LABELS.items():
        folder = os.path.join(DATA_DIR, label_name)
        if not os.path.isdir(folder):
            print(f"Warning: folder not found: {folder}")
            continue

        print(f"Loading '{label_name}' samples...")

        for file_name in os.listdir(folder):
            if not file_name.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                continue

            file_path = os.path.join(folder, file_name)
            features = extract_features(file_path)

            if features is not None:
                X.append(features)
                y.append(label_idx)

    return np.array(X), np.array(y)


# -----------------------------
# TRAIN MODEL
# -----------------------------
def train_model():
    print("Loading dataset...")
    X, y = load_dataset()

    if len(y) == 0:
        print("No training samples found!")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\nAccuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=LABELS.keys()))

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved as: {MODEL_PATH}")


# -----------------------------
# RUN TRAINING
# -----------------------------
if __name__ == "__main__":
    train_model()
