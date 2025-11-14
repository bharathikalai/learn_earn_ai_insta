from ultralytics import YOLO
import cv2
import numpy as np

# --- SETTINGS ---
video_path = "input.mp4"           # your input video
output_path = "output_with_speed.mp4"
model_path = "yolov8n.pt"          # pretrained model (detects vehicles)
scale_factor = 0.05                # meters per pixel (tune for your camera)

# Load YOLO model
model = YOLO(model_path)

# Read video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Simple tracker: remember previous center points
prev_centers = {}  # id -> (x, y)

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # Run detection
    results = model(frame, verbose=False)[0]
    detections = results.boxes.data.cpu().numpy()  # [x1,y1,x2,y2,score,class]

    new_centers = {}
    for i, det in enumerate(detections):
        x1, y1, x2, y2, conf, cls = det
        cls = int(cls)
        if cls not in [2, 3, 5, 7]:  # 2=car,3=motorbike,5=bus,7=truck (COCO classes)
            continue

        # center of box
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
        new_centers[i] = (cx, cy)

        # Speed estimation if same ID seen before
        speed_kmh = 0
        if i in prev_centers:
            prev_cx, prev_cy = prev_centers[i]
            pixel_dist = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
            meters_moved = pixel_dist * scale_factor
            speed_mps = meters_moved * fps
            speed_kmh = speed_mps * 3.6

        # Draw box and speed
        label = f"Vehicle {i} | {speed_kmh:.1f} km/h"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(frame, label, (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    prev_centers = new_centers
    out.write(frame)
    cv2.imshow("Speed Detection", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Saved:", output_path)
