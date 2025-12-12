import cv2
import torch
import numpy as np

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords

# -----------------------------------------------------------
# Load YOLO-CROWD model
# -----------------------------------------------------------
MODEL_PATH = "yolo-crowd.pt"
model = attempt_load(MODEL_PATH, map_location="cpu")
model.eval()

# -----------------------------------------------------------
# Process video
# -----------------------------------------------------------
INPUT_VIDEO  = "tvk.mp4"
OUTPUT_VIDEO = "output_heads.mp4"

cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)

# Original size (if needed)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# -------------------------------
# Medium output video resolution
# -------------------------------
OUT_W, OUT_H = 560, 840
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (OUT_W, OUT_H))

print("Processing video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------- Prepare input --------
    img = letterbox(frame, 640, stride=32)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img_tensor = torch.from_numpy(img).float()
    img_tensor /= 255.0
    img_tensor = img_tensor.unsqueeze(0)

    # -------- Run inference --------
    pred = model(img_tensor)[0]
    pred = non_max_suppression(pred, 0.25, 0.45)[0]

    head_count = 0

    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], frame.shape).round()

        for *xyxy, conf, cls in pred:
            x1, y1, x2, y2 = map(int, xyxy)

            # -------- HEAD region (top 30%) --------
            head_h = int((y2 - y1) * 0.40)
            hy2 = y1 + head_h

            cv2.rectangle(frame, (x1, y1), (x2, hy2), (0, 0, 255), 2)
            cv2.putText(frame, "head", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            head_count += 1

    # Display count
    cv2.putText(frame, f"Head Count: {head_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # -----------------------------------
    # Resize frame before writing & show
    # -----------------------------------
    resized_frame = cv2.resize(frame, (OUT_W, OUT_H))

    out.write(resized_frame)
    cv2.imshow("YOLO-CROWD Head Detection", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("DONE! Saved:", OUTPUT_VIDEO)
