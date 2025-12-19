import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load face detector
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# Load emotion model
emotion_model = load_model("emotion_model.h5")

# Emotion labels
emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

# 🎥 Input video
cap = cv2.VideoCapture("bharathi.mp4")

if not cap.isOpened():
    print("❌ Error: Video file not found")
    exit()

# 🎥 Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 🎬 Output video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    "bharathi_output.mp4",
    fourcc,
    fps,
    (width, height)
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (64, 64))
        roi_gray = roi_gray / 255.0
        roi_gray = roi_gray.reshape(1, 64, 64, 1)

        preds = emotion_model.predict(roi_gray, verbose=0)
        emotion = emotion_labels[np.argmax(preds)]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            emotion,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2
        )

    # 💾 Save frame
    out.write(frame)

    # 👀 Display (optional)
    cv2.imshow("Emotion Detection - Video", frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
