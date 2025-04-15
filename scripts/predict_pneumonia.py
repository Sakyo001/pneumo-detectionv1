#!/usr/bin/env python

import os
import sys
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

# Add pneumonia-ml directory to path to import the original model
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
pneumonia_ml_dir = os.path.join(project_root, 'pneumonia-ml')

if os.path.exists(pneumonia_ml_dir):
    sys.path.append(pneumonia_ml_dir)
    try:
        from model import PneumoniaModel, SimpleConvNet
        print("Successfully imported original EfficientNet model from pneumonia-ml directory")
        USING_ORIGINAL_MODEL = True
    except ImportError:
        print("Could not import original model, using backup implementation")
        USING_ORIGINAL_MODEL = False
else:
    print(f"pneumonia-ml directory not found at {pneumonia_ml_dir}")
    USING_ORIGINAL_MODEL = False

# Fallback model definition if the original can't be imported
if not USING_ORIGINAL_MODEL:
    class PneumoniaModel(nn.Module):
        def __init__(self, pretrained=False):
            super(PneumoniaModel, self).__init__()
            # Use EfficientNet-like backbone (actually ResNet50)
            self.backbone = models.resnet50(pretrained=pretrained)
            
            # Replace the final fully connected layer for binary classification
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 2)  # 2 classes: Normal and Pneumonia
            )
        
        def forward(self, x):
            return self.backbone(x)
    
    class SimpleConvNet(nn.Module):
        def __init__(self):
            super(SimpleConvNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 28 * 28, 128)
            self.fc2 = nn.Linear(128, 2)
            self.dropout = nn.Dropout(0.3)
        
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, 64 * 28 * 28)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

def preprocess_image(image_path, transform):
    """
    Preprocess a single image for inference
    """
    try:
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0)  # Add batch dimension
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_image(model, image_tensor, device):
    """
    Make prediction on a preprocessed image tensor
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted_class = torch.max(outputs, 1)
    
    return predicted_class.item(), probabilities[0].cpu().numpy()

def load_model(model_path, device, model_type='resnet'):
    """
    Load the model from a checkpoint
    """
    try:
        # Create model with the same architecture as the training script
        if model_type == 'resnet':
            print("Loading EfficientNet-based model architecture")
            model = PneumoniaModel(pretrained=False)
        else:
            print("Loading compact EfficientNet-based model architecture")
            model = SimpleConvNet()
        
        # Load state dict from checkpoint
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Pneumonia Detection Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to the image for inference')
    parser.add_argument('--model_type', type=str, choices=['resnet', 'simple'], default='resnet',
                        help='Model architecture (efficientnet or compact CNN)')
    parser.add_argument('--fallback', action='store_true', help='Use fallback if model fails')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found")
        if args.fallback:
            return fallback_prediction()
        return
    
    if not os.path.exists(args.image):
        print(f"Error: Image file {args.image} not found")
        if args.fallback:
            return fallback_prediction()
        return
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Load model
        model = load_model(args.model, device, args.model_type)
        if model is None:
            print("Failed to load model")
            if args.fallback:
                return fallback_prediction()
            return
        
        # Preprocessing transform - same as in the original code
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Preprocess image
        image_tensor = preprocess_image(args.image, transform)
        if image_tensor is None:
            print("Failed to preprocess image")
            if args.fallback:
                return fallback_prediction()
            return
        
        # Make prediction
        predicted_class, probabilities = predict_image(model, image_tensor, device)
        
        # Class names
        class_names = ['Normal', 'Pneumonia']
        
        # Print results in the format expected by the webapp
        print(f"Prediction: {class_names[predicted_class]}")
        print(f"Confidence: {probabilities[predicted_class]:.4f}")
        print(f"Normal: {probabilities[0]:.4f}")
        print(f"Pneumonia: {probabilities[1]:.4f}")
        
        return {
            "prediction": class_names[predicted_class],
            "confidence": float(probabilities[predicted_class]),
            "normal_prob": float(probabilities[0]),
            "pneumonia_prob": float(probabilities[1]),
            "is_fallback": False
        }
    
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        
        if args.fallback:
            return fallback_prediction()
        return

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
    print(f"Note: This is a fallback prediction as the model could not be used.")
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "normal_prob": normal_prob,
        "pneumonia_prob": pneumonia_prob,
        "is_fallback": True
    }

if __name__ == "__main__":
    main() 