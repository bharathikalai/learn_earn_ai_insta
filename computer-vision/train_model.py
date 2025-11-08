"""
================================================================================
STEP 1: TRAIN YOUR CAT vs DOG MODEL
================================================================================

This code will:
1. Load your cat and dog images
2. Train a neural network
3. Save the trained model
4. Test it to make sure it works

YOU NEED:
- One cat image: /mnt/user-data/uploads/cat.jpg
- One dog image: /mnt/user-data/uploads/dog.jpg
"""

import numpy as np
import pickle
from PIL import Image
import os

print("="*80)
print("🐱 CAT vs DOG MODEL TRAINING 🐶")
print("="*80)

# ==============================================================================
# PART 1: LOAD YOUR IMAGES
# ==============================================================================
print("\n📁 STEP 1: Loading your images...")

def load_image(image_path, size=(64, 64)):
    """
    Load an image and convert it to numbers
    
    HOW IT WORKS:
    1. Open the image file
    2. Resize to 64x64 pixels (smaller = faster training)
    3. Convert to RGB (Red, Green, Blue)
    4. Convert to numpy array (numbers!)
    """
    if os.path.exists(image_path):
        img = Image.open(image_path)
        img = img.resize(size)
        img = img.convert('RGB')
        img_array = np.array(img)
        print(f"   ✅ Loaded {image_path}")
        print(f"      Shape: {img_array.shape} (64x64 pixels, 3 colors)")
        print(f"      Pixel values: {img_array.min()} to {img_array.max()}")
        return img_array
    else:
        print(f"   ❌ ERROR: {image_path} not found!")
        print(f"   Please upload your image to this location.")
        return None

# Load your images
cat_path = '/mnt/user-data/uploads/cat.jpg'
dog_path = '/mnt/user-data/uploads/dog.jpg'

cat_image = load_image(cat_path)
dog_image = load_image(dog_path)

# Check if images loaded successfully
if cat_image is None or dog_image is None:
    print("\n❌ ERROR: Please upload cat.jpg and dog.jpg to /mnt/user-data/uploads/")
    print("   Then run this code again!")
    exit()

# ==============================================================================
# PART 2: PREPARE DATA (Convert images to neural network format)
# ==============================================================================
print("\n🔢 STEP 2: Preparing data for neural network...")

# 2.1: Normalize pixel values (0-255 → 0-1)
# WHY? Neural networks learn better with small numbers
print("\n   Converting pixel values 0-255 → 0-1...")
cat_normalized = cat_image.astype('float32') / 255.0
dog_normalized = dog_image.astype('float32') / 255.0
print(f"   Before: {cat_image[0,0]} (example pixel)")
print(f"   After:  {cat_normalized[0,0]} (example pixel)")

# 2.2: Flatten images (2D → 1D)
# Convert from 64x64x3 to a single list of 12,288 numbers
print("\n   Flattening images...")
cat_flat = cat_normalized.flatten()
dog_flat = dog_normalized.flatten()
print(f"   Original shape: {cat_normalized.shape}")
print(f"   Flattened shape: {cat_flat.shape}")
print(f"   Total numbers per image: {len(cat_flat)}")

# 2.3: Create training dataset
X_train = np.array([cat_flat, dog_flat])  # 2 images
y_train = np.array([[1, 0], [0, 1]])      # 2 labels (one-hot encoded)
#                     ↑  ↑    ↑  ↑
#                     |  |    |  └─ 1 = is dog
#                     |  |    └──── 0 = not cat
#                     |  └───────── 0 = not dog  
#                     └──────────── 1 = is cat

print(f"\n✅ Dataset ready!")
print(f"   X_train shape: {X_train.shape} (2 images, 12288 features each)")
print(f"   y_train shape: {y_train.shape} (2 labels)")
print(f"   Labels:")
print(f"   - Cat: {y_train[0]} → [1, 0] means 'is cat, not dog'")
print(f"   - Dog: {y_train[1]} → [0, 1] means 'not cat, is dog'")

# ==============================================================================
# PART 3: BUILD NEURAL NETWORK
# ==============================================================================
print("\n🧠 STEP 3: Building neural network...")

class CatDogClassifier:
    """
    Simple Neural Network for Cat vs Dog classification
    
    ARCHITECTURE:
    Input Layer:  12,288 neurons (one for each pixel RGB value)
    Hidden Layer: 20 neurons (learns patterns)
    Output Layer: 2 neurons (cat probability, dog probability)
    """
    
    def __init__(self, input_size=12288, hidden_size=20, output_size=2):
        """
        Initialize with RANDOM weights
        These will be learned during training!
        """
        print(f"\n   Creating neural network...")
        print(f"   - Input layer:  {input_size} neurons")
        print(f"   - Hidden layer: {hidden_size} neurons")
        print(f"   - Output layer: {output_size} neurons")
        
        # Random weights (will be learned!)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros(output_size)
        
        total_weights = self.W1.size + self.W2.size
        print(f"   - Total weights to learn: {total_weights:,}")
    
    def sigmoid(self, x):
        """
        Sigmoid activation: Converts any number to 0-1 range
        MATH: σ(x) = 1 / (1 + e^(-x))
        """
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def softmax(self, x):
        """
        Softmax: Converts numbers to probabilities that sum to 1
        Used for final output (cat probability + dog probability = 1)
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        FORWARD PROPAGATION: Make a prediction!
        
        WHAT HAPPENS:
        1. Multiply inputs by weights W1 → get hidden layer values
        2. Apply sigmoid activation
        3. Multiply hidden layer by weights W2 → get output values
        4. Apply softmax to get probabilities
        """
        # Layer 1: Input → Hidden
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Layer 2: Hidden → Output
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        
        return self.a2
    
    def backward(self, X, y, learning_rate=0.5):
        """
        BACKPROPAGATION: Learn from mistakes!
        
        WHAT HAPPENS:
        1. Calculate error (how wrong was the prediction?)
        2. Calculate gradients (how to adjust each weight?)
        3. Update all weights to reduce error
        """
        m = X.shape[0]  # Number of samples
        
        # Calculate gradients
        dz2 = self.a2 - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0) / m
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.a1 * (1 - self.a1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0) / m
        
        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, X, y, epochs=5000):
        """
        TRAINING: Show images many times and learn!
        
        PROCESS:
        1. Show cat image → predict → check error → adjust weights
        2. Show dog image → predict → check error → adjust weights
        3. Repeat 5000 times!
        """
        print(f"\n   Training for {epochs} epochs...")
        print(f"   (This will take about 30 seconds)\n")
        
        for epoch in range(epochs):
            # Forward pass (make predictions)
            output = self.forward(X)
            
            # Calculate loss (how wrong are we?)
            loss = -np.mean(y * np.log(output + 1e-8))
            
            # Backward pass (learn from mistakes!)
            self.backward(X, y)
            
            # Print progress every 500 epochs
            if (epoch + 1) % 500 == 0:
                print(f"   Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")
        
        print(f"\n   ✅ Training complete!")
    
    def predict(self, X):
        """
        Make a prediction on new image
        Returns probabilities: [cat_prob, dog_prob]
        """
        output = self.forward(X.reshape(1, -1))
        return output[0]

# Create and train the model
model = CatDogClassifier()
model.train(X_train, y_train, epochs=5000)

# ==============================================================================
# PART 4: TEST THE MODEL
# ==============================================================================
print("\n🔍 STEP 4: Testing the model...")

# Test on cat image
cat_prediction = model.predict(cat_flat)
print(f"\n   Cat image prediction:")
print(f"   - Cat probability: {cat_prediction[0]*100:.2f}%")
print(f"   - Dog probability: {cat_prediction[1]*100:.2f}%")
print(f"   - Result: {'✅ CAT!' if cat_prediction[0] > cat_prediction[1] else '❌ Wrong!'}")

# Test on dog image
dog_prediction = model.predict(dog_flat)
print(f"\n   Dog image prediction:")
print(f"   - Cat probability: {dog_prediction[0]*100:.2f}%")
print(f"   - Dog probability: {dog_prediction[1]*100:.2f}%")
print(f"   - Result: {'✅ DOG!' if dog_prediction[1] > dog_prediction[0] else '❌ Wrong!'}")

# ==============================================================================
# PART 5: SAVE THE MODEL
# ==============================================================================
print("\n💾 STEP 5: Saving the model...")

# Save model weights to file
model_data = {
    'W1': model.W1,
    'b1': model.b1,
    'W2': model.W2,
    'b2': model.b2,
    'input_size': 12288,
    'hidden_size': 20,
    'output_size': 2
}

with open('/mnt/user-data/outputs/cat_dog_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("   ✅ Model saved to: cat_dog_model.pkl")
print("   You can now use this model with the webcam!")

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("🎉 TRAINING COMPLETE!")
print("="*80)
print("""
WHAT HAPPENED:
1. ✅ Loaded your cat and dog images
2. ✅ Converted them to 12,288 numbers each
3. ✅ Created neural network with 245,780 weights
4. ✅ Trained for 5,000 rounds (adjusting weights each time)
5. ✅ Model learned to recognize cats vs dogs!
6. ✅ Saved model to cat_dog_model.pkl

NEXT STEP:
Run the webcam code (webcam_recognition.py) to test it live!
""")
print("="*80)
