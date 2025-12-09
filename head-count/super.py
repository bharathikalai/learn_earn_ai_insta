import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

# 1. Initialize the YOLO model
model = YOLO('yolov8n.pt') 

# 2. Initialize Annotators
# We skip complex annotation initialization entirely to avoid errors
print("Warning: Skipping advanced BoxAnnotator setup due to API error.")

# 3. Process the video
video_path = "TfiQRISGmcI_INBQ.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Processing video with dimensions: {frame_width}x{frame_height}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection (YOLOv8)
    results = model(frame, verbose=False)[0] 
    
    # Convert results to Supervision Detections format
    detections = sv.Detections.from_ultralytics(results)
    
    # Filter for 'person' class (class_id 0 in COCO dataset)
    detections = detections[detections.class_id == 0] 
    
    # --- COUNTING LOGIC ---
    current_count = len(detections)
    
    # --- ANNOTATION (Manual Bounding Boxes) ---
    annotated_frame = frame.copy()
    
    # Get bounding box coordinates from the detections object
    # xyxy is a numpy array of shape (N, 4) where N is the number of detections
    boxes = detections.xyxy
    
    # Manually draw bounding boxes using OpenCV
    for x1, y1, x2, y2 in boxes:
        # Convert coordinates to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw the rectangle (Box color: Blue (255, 0, 0))
        cv2.rectangle(
            img=annotated_frame, 
            pt1=(x1, y1), 
            pt2=(x2, y2), 
            color=(255, 0, 0), 
            thickness=2
        )
    
    # Display the final count prominently on the frame
    cv2.putText(
        img=annotated_frame,
        text=f"Total Count: {current_count}",
        org=(50, 50), 
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.5,
        color=(0, 255, 0), # Green color
        thickness=3
    )

    cv2.imshow("In-Frame Crowd Count", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()