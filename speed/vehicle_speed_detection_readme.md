# Vehicle Speed Detection using YOLOv8

This project processes a video, detects vehicles using a YOLO model, tracks their movement across frames, estimates their speed, overlays the speed onto the video, and saves the processed output.

This README explains everything in **very simple language**, step-by-step, so you fully understand how the code works.

---

# 📌 What This Program Does
1. Loads a YOLO model that can detect vehicles.
2. Reads an input video.
3. Detects vehicles in each frame.
4. Tracks how the vehicle moves across frames.
5. Calculates speed using how many pixels it moved.
6. Converts pixel movement to real-world speed (km/h).
7. Draws the speed text on the video.
8. Saves a new video you can download.

---

# 📦 Requirements
Install dependencies:
```bash
pip install ultralytics opencv-python numpy
```

Files you need:
- `main.py` (your program)
- `yolov8n.pt` (pretrained YOLO model)
- `input.mp4` (your input road video)

---

# 📁 Project Structure
```
vehical-speed-detected/
│── main.py
│── yolov8n.pt
│── input.mp4
│── output_with_speed.mp4 (created after running)
│── README.md
```

---

# 🧠 How Speed Detection Works (Simple Explanation)
A video is just many images shown quickly.

For each image (frame):
- We detect vehicles.
- We find the center of the vehicle box.
- We compare this center with the previous frame's center.
- The distance between these two points tells us how far the vehicle moved.
- Using FPS and a scale factor, we convert movement → speed.

**Formula:**
```
pixels_moved → meters_moved → meters_per_second → km/h
```

---

# 🔢 COCO Class IDs Used
The YOLO model detects 80 object types. But we only care about vehicle classes:

| Class ID | Object        |
|----------|----------------|
| 1        | Bicycle        |
| 2        | Car            |
| 3        | Motorbike      |
| 5        | Bus            |
| 7        | Truck          |

So inside the code:
```python
if cls not in [1, 2, 3, 5, 7]:
    continue
```
This ensures we only detect vehicles.

---

# 🧩 Full Code Explanation (main.py)
Below is the full code with simple explanation for each section.

---

## 1. Imports
```python
from ultralytics import YOLO
import cv2
import numpy as np
```
- `YOLO` → loads the YOLOv8 model
- `cv2` → reads/writes video, draws boxes
- `numpy` → does math

---

## 2. Settings
```python
video_path = "input.mp4"
output_path = "output_with_speed.mp4"
model_path = "yolov8n.pt"
scale_factor = 0.05
```
- `video_path` → your input video
- `output_path` → saved output
- `model_path` → YOLO model file
- `scale_factor` → meters per pixel (you can tune this)

---

## 3. Load YOLO model
```python
model = YOLO(model_path)
```
YOLO detects vehicles in each frame.

---

## 4. Open the video
```python
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
```
We extract:
- FPS → needed for speed calculation
- Width & height → needed to write the output video

---

## 5. Create output video writer
```python
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
```
This saves each processed frame to a new video file.

---

## 6. Simple Tracker
```python
prev_centers = {}  # id -> (x, y)
```
This remembers where each vehicle was in the last frame.

---

## 7. Main Loop — frame by frame
```python
while True:
    ret, frame = cap.read()
    if not ret:
        break
```
Reads each frame until video ends.

---

## 8. Run YOLO Detection
```python
results = model(frame, verbose=False)[0]
detections = results.boxes.data.cpu().numpy()\```
YOLO gives bounding boxes like:
```
[x1, y1, x2, y2, score, class_id]
```

---

## 9. Filter only vehicles
```python
if cls not in [1, 2, 3, 5, 7]:
    continue
```
Only detect:
- Bicycles
- Cars
- Motorbikes
- Buses
- Trucks

---

## 10. Find box center
```python
cx, cy = int((x1+x2)/2), int((y1+y2)/2)
```
This gives the middle point of the vehicle.

---

## 11. Speed Calculation
```python
if i in prev_centers:
    prev_cx, prev_cy = prev_centers[i]
    pixel_dist = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)

    meters_moved = pixel_dist * scale_factor
    speed_mps = meters_moved * fps
    speed_kmh = speed_mps * 3.6
```
Steps:
1. Compare current center with last frame center.
2. Get how many pixels it moved.
3. Convert pixels → meters.
4. Convert meters → speed.

---

## 12. Draw Everything
```python
label = f"Vehicle {i} | {speed_kmh:.1f} km/h"
cv2.rectangle(frame, ...)
cv2.putText(frame, ...)
```
Draws:
- green box
- vehicle number
- estimated speed

---

## 13. Save Output
```python
prev_centers = new_centers
out.write(frame)
```
Each processed frame is added to the new video.

---

## 14. Optional Display
```python
cv2.imshow("Speed Detection", frame)
if cv2.waitKey(1) == 27:
    break
```
Shows video in a window (ESC to quit).

If this crashes on Linux, comment it out.

---

## 15. Cleanup
```python
cap.release()
out.release()
cv2.destroyAllWindows()
```
Closes everything properly.

---

# 🚀 Result
A new video file will be created:
```
output_with_speed.mp4
```
This video will contain:
- Bounding boxes around vehicles
- Speed displayed for each one
- Frames stitched together in MP4

---

# 🎯 Next Improvements (Optional)
- Use DeepSORT for better vehicle tracking
- Calibrate correct scale factor using real road measurements
- Add different colors for each vehicle type
- Export detection logs as CSV

---

# 💬 Need More Help?
Ask me to:
- Add DeepSORT tracking
- Add lane detection
- Add distance measurement between vehicles
- Make a GUI for uploading videos

I’m here to help you build the full proj