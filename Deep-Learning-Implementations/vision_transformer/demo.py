#!/usr/bin/env python3
"""
Demo script for Vision Transformer inference
"""

import torch
import torch.nn.functional as F
import argparse
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from model import DawidGPT, create_patches

# CIFAR-10 classes
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def load_image(image_path, device):
    """Load and preprocess image"""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Create patches for the model
    patches = create_patches(image_tensor, patch_size=4)
    patches = patches.squeeze(0)  # Remove batch dimension for single image
    
    return patches, image

def predict(model, patches, device):
    """Make prediction on image patches"""
    model.eval()
    with torch.no_grad():
        # Add batch dimension
        patches = patches.unsqueeze(0).to(device)
        outputs = model(patches)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        return predicted.item(), confidence.item(), probabilities.squeeze()

def main():
    parser = argparse.ArgumentParser(description='Vision Transformer Demo')
    parser.add_argument('--image', type=str, required=True, 
                       help='Path to input image')
    parser.add_argument('--model_path', type=str, default='best_vit_model.pt', 
                       help='Path to trained model')
    parser.add_argument('--top_k', type=int, default=5, 
                       help='Show top k predictions')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    try:
        model = DawidGPT(num_classes=10)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Model file not found! Please train the model first using train.py")
        return
    
    # Load and process image
    try:
        patches, original_image = load_image(args.image, device)
        print(f"Image loaded: {args.image}")
        print(f"Patches shape: {patches.shape}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Make prediction
    try:
        predicted_class, confidence, probabilities = predict(model, patches, device)
    except Exception as e:
        print(f"Error making prediction: {e}")
        return
    
    # Display results
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(f"Predicted Class: {CIFAR10_CLASSES[predicted_class]}")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    
    # Show top k predictions
    print(f"\nTop {args.top_k} predictions:")
    top_k_probs, top_k_indices = torch.topk(probabilities, args.top_k)
    
    for i, (prob, idx) in enumerate(zip(top_k_probs, top_k_indices)):
        class_name = CIFAR10_CLASSES[idx.item()]
        print(f"{i+1}. {class_name}: {prob.item():.4f} ({prob.item()*100:.2f}%)")
    
    print("="*50)

if __name__ == "__main__":
    main()
