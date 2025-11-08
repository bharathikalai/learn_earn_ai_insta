"""
================================================================================
STEP 2: REAL-TIME CAT vs DOG RECOGNITION WITH WEBCAM
================================================================================

This code will:
1. Load your trained model
2. Open your webcam
3. Show what the camera sees
4. Recognize if it's a cat or dog in REAL-TIME!

REQUIREMENTS:
- You must run train_model.py first!
- You need a webcam
- Model file: cat_dog_model.pkl must exist
"""

import numpy as np
import pickle
import cv2
from PIL import Image
import os

print("="*80)
print("🎥 REAL-TIME CAT vs DOG RECOGNITION")
print("="*80)

# ==============================================================================
# PART 1: LOAD THE TRAINED MODEL
# ==============================================================================
print("\n📂 STEP 1: Loading trained model...")

model_path = '/mnt/user-data/outputs/cat_dog_model.pkl'

if not os.path.exists(model_path):
    print(f"\n❌ ERROR: Model not found at {model_path}")
    print("   Please run train_model.py first!")
    exit()

# Load model weights
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

# Extract weights
W1 = model_data['W1']
b1 = model_data['b1']
W2 = model_data['W2']
b2 = model_data['b2']

print("✅ Model loaded successfully!")
print(f"   Total weights: {W1.size + W2.size:,}")

# ==============================================================================
# PART 2: DEFINE PREDICTION FUNCTION
# ==============================================================================

def sigmoid(x):
    """
    Sigmoid activation function
    Converts any number to 0-1 range
    """
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def softmax(x):
    """
    Softmax function
    Converts numbers to probabilities that sum to 1
    """
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def predict_image(image_array):
    """
    Predict if image is cat or dog
    
    INPUT: Image as numpy array (any size)
    OUTPUT: [cat_probability, dog_probability]
    
    STEPS:
    1. Resize image to 64x64 (same as training!)
    2. Normalize to 0-1
    3. Flatten to 12,288 numbers
    4. Pass through neural network
    5. Return probabilities
    """
    # Step 1: Resize to 64x64
    img = Image.fromarray(image_array)
    img = img.resize((64, 64))
    img_array = np.array(img)
    
    # Step 2: Normalize
    img_normalized = img_array.astype('float32') / 255.0
    
    # Step 3: Flatten
    img_flat = img_normalized.flatten()
    
    # Step 4: Forward propagation through network
    # Layer 1: Input → Hidden
    z1 = np.dot(img_flat, W1) + b1
    a1 = sigmoid(z1)
    
    # Layer 2: Hidden → Output
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)
    
    # Step 5: Return probabilities
    return a2

# ==============================================================================
# PART 3: OPEN WEBCAM AND START RECOGNITION
# ==============================================================================
print("\n🎥 STEP 2: Opening webcam...")
print("\nINSTRUCTIONS:")
print("- Show a cat or dog picture/toy to the camera")
print("- The model will tell you if it's a cat or dog")
print("- Press 'q' to quit")
print("- Press 's' to save a snapshot")
print("\nStarting in 3 seconds...\n")

import time
time.sleep(3)

# Open webcam
# 0 = default webcam, change to 1 if you have multiple cameras
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: Cannot open webcam!")
    print("   Please check if your webcam is connected.")
    exit()

print("✅ Webcam opened successfully!")
print("=" * 80)

frame_count = 0

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    
    if not ret:
        print("❌ ERROR: Cannot read from webcam!")
        break
    
    frame_count += 1
    
    # Only predict every 5 frames (to avoid lag)
    if frame_count % 5 == 0:
        try:
            # Convert BGR (OpenCV format) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Make prediction
            probabilities = predict_image(frame_rgb)
            cat_prob = probabilities[0]
            dog_prob = probabilities[1]
            
            # Determine result
            if cat_prob > dog_prob:
                label = "CAT"
                confidence = cat_prob * 100
                color = (255, 0, 255)  # Magenta for cat
            else:
                label = "DOG"
                confidence = dog_prob * 100
                color = (0, 255, 255)  # Yellow for dog
            
            # Draw results on frame
            # Background rectangle for text
            cv2.rectangle(frame, (10, 10), (400, 120), (0, 0, 0), -1)
            
            # Main label
            cv2.putText(frame, f"{label}", 
                       (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       2, color, 3)
            
            # Confidence
            cv2.putText(frame, f"Confidence: {confidence:.1f}%", 
                       (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
            
            # Probabilities bar
            bar_width = 300
            cat_bar_length = int(cat_prob * bar_width)
            dog_bar_length = int(dog_prob * bar_width)
            
            # Cat probability bar
            cv2.rectangle(frame, (10, 140), (10 + bar_width, 160), (100, 100, 100), -1)
            cv2.rectangle(frame, (10, 140), (10 + cat_bar_length, 160), (255, 0, 255), -1)
            cv2.putText(frame, f"Cat: {cat_prob*100:.1f}%", 
                       (320, 155), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)
            
            # Dog probability bar
            cv2.rectangle(frame, (10, 170), (10 + bar_width, 190), (100, 100, 100), -1)
            cv2.rectangle(frame, (10, 170), (10 + dog_bar_length, 190), (0, 255, 255), -1)
            cv2.putText(frame, f"Dog: {dog_prob*100:.1f}%", 
                       (320, 185), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)
            
            # Instructions
            cv2.putText(frame, "Press 'q' to quit | Press 's' to save", 
                       (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)
            
        except Exception as e:
            print(f"⚠️  Prediction error: {e}")
    
    # Show frame
    cv2.imshow('Cat vs Dog Recognition', frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        # Quit
        print("\n👋 Closing webcam...")
        break
    elif key == ord('s'):
        # Save snapshot
        filename = f'/mnt/user-data/outputs/snapshot_{int(time.time())}.jpg'
        cv2.imwrite(filename, frame)
        print(f"📸 Snapshot saved: {filename}")

# Clean up
cap.release()
cv2.destroyAllWindows()

print("\n✅ Webcam closed!")
print("="*80)
print("Thanks for using Cat vs Dog Recognition!")
print("="*80)
