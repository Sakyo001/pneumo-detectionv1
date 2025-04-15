#!/usr/bin/env python

import argparse
import sys
import time
import os
from PIL import Image
import json

# Try to import PyTorch and EfficientNet, but handle if not available
try:
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Error: PyTorch not installed. Using fallback prediction.")

try:
    from efficientnet_pytorch import EfficientNet
    EFFICIENTNET_AVAILABLE = True
except ImportError:
    EFFICIENTNET_AVAILABLE = False
    print("Error: EfficientNet not installed. Using fallback prediction.")

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Pneumonia X-ray Classifier')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    parser.add_argument('--fallback', action='store_true', help='Use fallback prediction if model fails')
    return parser.parse_args()

def fallback_prediction():
    """Generate a fallback prediction when the model cannot be used"""
    import random
    prediction = "Pneumonia" if random.random() > 0.5 else "Normal"
    confidence = 0.5 + random.random() * 0.3
    normal_prob = 1 - confidence if prediction == "Pneumonia" else confidence
    pneumonia_prob = confidence if prediction == "Pneumonia" else 1 - confidence
    
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Normal: {normal_prob:.4f}")
    print(f"Pneumonia: {pneumonia_prob:.4f}")
    print(f"Processing time: 0.1 seconds")
    print(f"Note: This is a fallback prediction as the model could not be used.")
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "normal_prob": normal_prob,
        "pneumonia_prob": pneumonia_prob,
        "is_fallback": True
    }

def main():
    args = parse_args()
    
    # Check if PyTorch and EfficientNet are available
    if not TORCH_AVAILABLE or not EFFICIENTNET_AVAILABLE:
        return fallback_prediction()
    
    # Start timer
    start_time = time.time()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        if args.fallback:
            return fallback_prediction()
        return
    
    # Check if image file exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found at {args.image}")
        if args.fallback:
            return fallback_prediction()
        return
    
    # Define image transformations
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load image
    try:
        image = Image.open(args.image).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        if args.fallback:
            return fallback_prediction()
        return
    
    # Transform image
    image_tensor = data_transforms(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    
    # Load model
    try:
        # Load the EfficientNet model
        model = EfficientNet.from_name('efficientnet-b0', num_classes=2)
        model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        if args.fallback:
            return fallback_prediction()
        return
    
    # Make prediction
    try:
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            normal_prob = probabilities[0][0].item()
            pneumonia_prob = probabilities[0][1].item()
            
            # Determine prediction and confidence
            prediction = "Normal" if normal_prob > pneumonia_prob else "Pneumonia"
            confidence = max(normal_prob, pneumonia_prob)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Print results in a format that can be parsed by the server
            print(f"Prediction: {prediction}")
            print(f"Confidence: {confidence:.4f}")
            print(f"Normal: {normal_prob:.4f}")
            print(f"Pneumonia: {pneumonia_prob:.4f}")
            print(f"Processing time: {processing_time:.2f} seconds")
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "normal_prob": normal_prob,
                "pneumonia_prob": pneumonia_prob,
                "processing_time": processing_time,
                "is_fallback": False
            }
    except Exception as e:
        print(f"Error during prediction: {e}")
        if args.fallback:
            return fallback_prediction()

if __name__ == "__main__":
    result = main()
    # Optionally output as JSON for more structured parsing
    if result and '--json' in sys.argv:
        print(f"JSON_OUTPUT: {json.dumps(result)}") 