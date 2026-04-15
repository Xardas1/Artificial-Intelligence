#!/usr/bin/env python3
"""
Demo script for NumPy CNN implementation
"""

import numpy as np
from PIL import Image
import argparse
import os

from model import forward_pass, create_sample_image

def load_and_preprocess_image(image_path, target_size=(32, 32)):
    """Load and preprocess image for CNN"""
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Resize to target size
        image = image.resize(target_size)
        
        # Convert to numpy array and normalize
        image_array = np.array(image) / 255.0
        
        return image_array
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def create_sample_image():
    """Create a sample image for demonstration"""
    # Create a simple gradient image
    height, width = 32, 32
    image = np.zeros((height, width, 3))
    
    # Create RGB gradient
    for i in range(height):
        for j in range(width):
            image[i, j, 0] = i / height  # Red gradient
            image[i, j, 1] = j / width   # Green gradient
            image[i, j, 2] = (i + j) / (height + width)  # Blue gradient
    
    return image

def predict_image(image_array):
    """Make prediction on image"""
    try:
        # Create random biases for the network
        bias1 = np.random.rand(120, 1)
        bias2 = np.random.rand(10, 1)
        
        # Forward pass through network
        predictions = forward_pass(image_array, bias1, bias2)
        
        return predictions
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

def display_predictions(predictions, class_names=None):
    if predictions is None:
        return
    
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(len(predictions))]
    
    print("\n" + "="*50)
    print("CNN PREDICTION RESULTS")
    print("="*50)
    
    sorted_indices = np.argsort(predictions.flatten())[::-1]
    
    print("Top 5 predictions:")
    for i in range(min(5, len(sorted_indices))):
        idx = sorted_indices[i]
        prob = predictions.flatten()[idx]
        class_name = class_names[idx] if idx < len(class_names) else f"Class_{idx}"
        print(f"{i+1}. {class_name}: {prob:.6f} ({prob*100:.2f}%)")
    
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description='NumPy CNN Demo')
    parser.add_argument('--image', type=str, 
                       help='Path to input image (optional, will create sample if not provided)')
    parser.add_argument('--size', type=int, default=32, 
                       help='Image size for processing (default: 32)')
    parser.add_argument('--classes', type=int, default=10, 
                       help='Number of output classes (default: 10)')
    
    args = parser.parse_args()
    
    print("NumPy CNN Demo")
    print("="*30)
    
    # Initialize model
    print(f"Initializing CNN with {args.classes} output classes...")
    model = NumpyCNN(input_shape=(args.size, args.size, 3), num_classes=args.classes)
    print("Model initialized successfully!")
    
    # Load or create image
    if args.image and os.path.exists(args.image):
        print(f"Loading image: {args.image}")
        image_array = load_and_preprocess_image(args.image, (args.size, args.size))
    else:
        print("Creating sample gradient image...")
        image_array = create_sample_image()
        if args.image:
            print(f"Warning: Could not load {args.image}, using sample image instead")
    
    if image_array is None:
        print("Failed to load/create image!")
        return
    
    print(f"Image shape: {image_array.shape}")
    
    # Make prediction
    try:
        predictions = predict_image(image_array)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return
    
    if predictions is not None:
        print(f"Prediction completed! Output shape: {predictions.shape}")
        
        # Display results
        display_predictions(predictions)
        
        # Additional info
        print(f"\nModel Architecture:")
        print(f"- Input: {image_array.shape}")
        print(f"- Conv1: 32 filters, 3x3")
        print(f"- MaxPool1: 2x2")
        print(f"- Conv2: 64 filters, 3x3") 
        print(f"- MaxPool2: 2x2")
        print(f"- FC1: 120 neurons")
        print(f"- FC2: {args.classes} neurons")
        print(f"- Output: Softmax")
        
        print(f"\nNote: This is a forward-pass only demo with random weights.")
        print(f"For actual classification, the model needs to be trained with backpropagation.")
    else:
        print("Prediction failed!")

if __name__ == "__main__":
    main()
