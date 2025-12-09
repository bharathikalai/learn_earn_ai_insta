import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

# --- CONFIGURATION ---
VIDEO_PATH = "TfiQRISGmcI_INBQ.mp4"
MODEL_WEIGHTS = 'yolov8n.pt' 
PERSON_CLASS_ID = 0
# We assume the head is the top 30% of the full person bounding box
HEAD_CROP_RATIO = 0.3 

# --- INITIALIZATION ---
try:
    # Initialize the YOLO model
    model = YOLO(MODEL_WEIGHTS)
except Exception as e:
    print(f"Error initializing model: {e}")
    print("Please ensure 'ultralytics' is installed.")
    exit()

# --- DETECTION AND ANNOTATION LOGIC (CORRECTED) ---
def detect_and_annotate_heads(frame: np.ndarray, model: YOLO):
    """Detects people, crops the box to the head, and draws manually."""
    
    # ... (Steps 1-3 remain the same) ...
    
    # 1. Run Detection
    results = model(frame, verbose=False)[0] 
    
    # 2. Convert results to Supervision Detections format
    detections = sv.Detections.from_ultralytics(results)
    
    # 3. Filter for 'person' only
    detections = detections[detections.class_id == PERSON_CLASS_ID] 
    
    # --- SIMULATE HEAD DETECTION by cropping the bounding box ---
    person_boxes = detections.xyxy
    
    # Calculate the height of the full person bounding box
    heights = person_boxes[:, 3] - person_boxes[:, 1]
    
    # Calculate the new y-coordinate for the bottom of the head box
    new_y2 = person_boxes[:, 1] + (heights * HEAD_CROP_RATIO)

    # We modify the array in place, but must still cast to int later for cv2
    head_boxes = person_boxes.copy()
    head_boxes[:, 3] = new_y2.astype(int) 
    
    # Get confidence scores for manual labeling
    confidences = detections.confidence
    
    # 4. Count
    current_count = len(detections)
    
    # 5. Manual Annotation
    annotated_frame = frame.copy()
    
    # Iterate through the cropped head boxes and confidence scores
    for i, (x1, y1, x2, y2) in enumerate(head_boxes):
        
        # *** THE CRITICAL FIX IS HERE ***
        # Convert all coordinates to native Python int() before passing to cv2
        x1_int, y1_int, x2_int, y2_int = int(x1), int(y1), int(x2), int(y2)
        
        # Draw the rectangle (Box color: Yellow/Cyan (255, 255, 0))
        cv2.rectangle(
            img=annotated_frame, 
            pt1=(x1_int, y1_int), 
            pt2=(x2_int, y2_int), 
            color=(255, 255, 0), 
            thickness=2
        )
        
        # Add a text label
        label_text = f"Head: {confidences[i]:.2f}"
        cv2.putText(
            img=annotated_frame,
            text=label_text,
            # Ensure text position coordinates are also integers
            org=(x1_int, y1_int - 10), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(255, 255, 0),
            thickness=1
        )
    
    return annotated_frame, current_count

# The rest of your main script remains the same, but ensure you replace
# the old function definition with this corrected one in your 'new.py' file.

# --- MAIN VIDEO LOOP ---
cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run the head detection simulation (no annotator object passed)
    annotated_frame, current_count = detect_and_annotate_heads(frame, model)

    # Display the total count prominently
    cv2.putText(
        img=annotated_frame,
        text=f"Estimated Heads: {current_count}",
        org=(50, 50), 
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.5,
        color=(0, 255, 255), # Yellow/Cyan color
        thickness=3
    )

    cv2.imshow("Top-View Head Detection Simulation (Manual Draw)", annotated_frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()