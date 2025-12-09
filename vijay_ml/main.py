import cv2
from ultralytics import YOLO

# --- Configuration ---
# 1. Path to your trained model weights.
#    - If you fine-tuned a model for 'head' detection, use that path (e.g., 'runs/detect/train/weights/best.pt').
#    - For this example, we'll use a standard YOLO model to detect 'person' as a proxy.
MODEL_PATH = 'yolov8n.pt'  # 'n' for nano, a fast and lightweight model

# 2. Path to your input video file
VIDEO_PATH = 'TfiQRISGmcI_INBQ.mp4'

# 3. Output video path
OUTPUT_PATH = 'output_head_count.avi'

# --- Load Model ---
# Load your custom or pre-trained YOLO model
model = YOLO(MODEL_PATH)

# --- Video Processing Function ---
def process_video_for_counting(video_path, model_path, output_path):
    """
    Processes a video, detects objects (e.g., heads/people), and displays the count.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties for saving the output
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use 'mp4v' for .mp4, 'XVID' for .avi
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print("Processing video... Press 'q' to stop.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO inference on the frame
        # We specify 'person' (class 0 in COCO) or your 'head' class if custom trained
        # Note: If your model is custom-trained for 'head', you don't need the 'classes' argument
        results = model(frame, classes=0, verbose=False) 

        # --- Count Detected Objects ---
        # The 'results' object contains all detection data
        # results[0].boxes.data is a tensor where each row is [x1, y1, x2, y2, confidence, class_id]
        detected_objects = results[0].boxes.data.cpu().numpy()
        
        # Filter for a specific class (optional if custom model is head-only)
        # Assuming class 0 is 'person' (COCO) or 'head' (custom dataset)
        head_count = len(detected_objects) 

        # --- Draw Results ---
        # The 'plot()' method draws bounding boxes and labels on the frame
        annotated_frame = results[0].plot()

        # Add the count text to the frame
        cv2.putText(
            annotated_frame,
            f"People/Heads Count: {head_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 0, 255),  # Red color
            2
        )

        # Write the annotated frame to the output video
        out.write(annotated_frame)
        
        # Display the frame
        cv2.imshow("Crowd Counting", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything when the job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Output saved to {output_path}")

# Run the function (make sure 'input_crowd_video.mp4' exists)
# process_video_for_counting(VIDEO_PATH, MODEL_PATH, OUTPUT_PATH)