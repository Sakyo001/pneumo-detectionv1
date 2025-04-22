#!/usr/bin/env python3
"""
Inference script for pneumonia detection model
Can work with various model types and formats
"""

import os
import sys
import json
import argparse
import time
import numpy as np
from datetime import datetime
import random
import hashlib
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from model import PneumoniaModel, SimpleConvNet

def preprocess_image(image_path, transform):
    """
    Preprocess a single image for inference
    """
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

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

def visualize_prediction(image_path, predicted_class, probabilities, class_names, output_path=None):
    """
    Visualize the prediction with the original image
    """
    image = Image.open(image_path).convert('RGB')
    
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original X-ray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    bars = plt.bar(class_names, probabilities)
    bars[predicted_class].set_color('red')
    plt.title(f'Prediction: {class_names[predicted_class]}')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

def apply_gradcam(model, image_tensor, class_idx, layer_name='layer4'):
    """
    Apply Grad-CAM to visualize areas contributing to the prediction
    """
    import cv2
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    
    # For ResNet50
    target_layers = [getattr(model.backbone, layer_name)]
    
    # Initialize GradCAM
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
    
    # Generate CAM
    grayscale_cam = cam(input_tensor=image_tensor, target_category=class_idx)
    grayscale_cam = grayscale_cam[0, :]
    
    # Convert tensor to numpy for visualization
    image_numpy = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    image_numpy = (image_numpy - image_numpy.min()) / (image_numpy.max() - image_numpy.min())
    
    # Overlay CAM on original image
    visualization = show_cam_on_image(image_numpy, grayscale_cam, use_rgb=True)
    
    return visualization

def batch_inference(model, data_dir, output_dir, model_type, batch_size=32):
    """
    Run inference on a directory of images
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Preprocessing transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Get list of image files
    image_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        print(f"No image files found in {data_dir}")
        return
    
    print(f"Found {len(image_files)} images for inference")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    # Process images in batches to speed up inference
    num_batches = (len(image_files) + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i+1) * batch_size, len(image_files))
        batch_files = image_files[start_idx:end_idx]
        
        # Prepare batch
        batch_tensors = []
        for img_path in batch_files:
            img_tensor = preprocess_image(img_path, transform)
            batch_tensors.append(img_tensor)
        
        batch_tensor = torch.cat(batch_tensors, dim=0)
        batch_tensor = batch_tensor.to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(batch_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted_classes = torch.max(outputs, 1)
        
        # Store results
        for j, img_path in enumerate(batch_files):
            pred_class = predicted_classes[j].item()
            probs = probabilities[j].cpu().numpy()
            
            result = {
                'image_path': img_path,
                'predicted_class': pred_class,
                'normal_probability': probs[0],
                'pneumonia_probability': probs[1],
                'prediction': 'Normal' if pred_class == 0 else 'Pneumonia'
            }
            
            results.append(result)
        
        print(f"Processed batch {i+1}/{num_batches}")
    
    # Save results to CSV
    import pandas as pd
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'inference_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Summary statistics
    pneumonia_count = sum(1 for r in results if r['predicted_class'] == 1)
    normal_count = len(results) - pneumonia_count
    
    print(f"\nInference Summary:")
    print(f"Total images: {len(results)}")
    print(f"Normal predictions: {normal_count} ({normal_count/len(results)*100:.1f}%)")
    print(f"Pneumonia predictions: {pneumonia_count} ({pneumonia_count/len(results)*100:.1f}%)")

# Define a function to create deterministic results based on a seed
def get_deterministic_result(image_path, reference_number=None):
    """
    Generate deterministic prediction results based on image path or reference number
    """
    # Use reference number as seed if available, otherwise use image path
    seed_value = reference_number if reference_number else image_path
    
    # Create a hash from the seed value
    hash_object = hashlib.md5(seed_value.encode())
    hash_hex = hash_object.hexdigest()
    
    # Convert first 8 chars of hash to an integer
    hash_int = int(hash_hex[:8], 16)
    
    # Seed the random number generator for deterministic output
    random.seed(hash_int)
    
    # Generate deterministic prediction
    confidence = 60 + (hash_int % 30) / 100.0  # Between 0.6 and 0.9
    is_pneumonia = hash_int % 2 == 0  # Even hash = pneumonia, odd = normal
    
    if is_pneumonia:
        # Pneumonia case
        pneumonia_type = "Bacterial" if hash_int % 3 == 0 else "Viral"
        severity = "Severe" if confidence > 0.85 else "Moderate" if confidence > 0.75 else "Mild"
        
        result = {
            "diagnosis": "Pneumonia",
            "confidence": round(confidence * 100),
            "pneumonia_type": pneumonia_type,
            "severity": severity,
            "usingMock": True,
            "processingTime": 1.2,
            "probabilities": {
                "normal": round((1.0 - confidence) * 100),
                "pneumonia": round(confidence * 100)
            }
        }
    else:
        # Normal case
        result = {
            "diagnosis": "Normal",
            "confidence": round(confidence * 100),
            "usingMock": True,
            "processingTime": 1.2,
            "probabilities": {
                "normal": round(confidence * 100),
                "pneumonia": round((1.0 - confidence) * 100)
            }
        }
    
    if is_pneumonia:
        if confidence > 0.85:
            result["recommendedAction"] = "Immediate medical attention recommended."
            result["severityDescription"] = "Severe pneumonia detected with high confidence."
        elif confidence > 0.75:
            result["recommendedAction"] = "Consult with a healthcare provider soon."
            result["severityDescription"] = "Moderate pneumonia detected."
        else:
            result["recommendedAction"] = "Monitor symptoms and rest. Consult a doctor if symptoms worsen."
            result["severityDescription"] = "Mild pneumonia detected."
    else:
        result["recommendedAction"] = "No pneumonia detected. Maintain healthy habits."
    
    return result

def main():
    """
    Main function for inference script
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run inference with pneumonia detection model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model file')
    parser.add_argument('--image_path', type=str, required=True, help='Path to image file')
    parser.add_argument('--model_type', type=str, default='cnn', help='Model type (cnn, resnet, efficientnet, pytorch)')
    parser.add_argument('--reference_number', type=str, default=None, help='Patient reference number')
    parser.add_argument('--patient_name', type=str, default=None, help='Patient name')
    parser.add_argument('--patient_age', type=str, default=None, help='Patient age')
    parser.add_argument('--patient_gender', type=str, default=None, help='Patient gender')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print("Model file not found, using mock prediction")
        result = get_deterministic_result(args.image_path, args.reference_number)
        print(json.dumps(result, indent=2))
        return
    
    # Check if image file exists
    if not os.path.exists(args.image_path):
        print("Image file not found")
        result = {
            "error": "Image file not found",
            "usingMock": True
        }
        print(json.dumps(result, indent=2))
        return
    
    try:
        # Try to import deep learning libraries - these may fail if not installed
        try:
            import torch
            import torchvision
            HAS_TORCH = True
        except ImportError:
            print("PyTorch not found, using mock prediction")
            HAS_TORCH = False
            
        # If torch is not available, use mock prediction
        if not HAS_TORCH:
            result = get_deterministic_result(args.image_path, args.reference_number)
            print(json.dumps(result, indent=2))
            return
            
        # Start timing
        start_time = time.time()
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load model based on type
        if args.model_type.lower() in ['pytorch', 'pth']:
            try:
                # Load PyTorch model
                model = PneumoniaModel(pretrained=False, freeze_backbone=False)
                # Handle loading with map_location to ensure it loads on the correct device
                state_dict = torch.load(args.model_path, map_location=device)
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()
                print(f"Loaded PyTorch model from {args.model_path}")
            except Exception as model_error:
                print(f"Error loading model: {model_error}")
                print("Trying alternative model loading approach...")
                try:
                    # Try loading with SimpleConvNet
                    model = SimpleConvNet()
                    state_dict = torch.load(args.model_path, map_location=device)
                    model.load_state_dict(state_dict)
                    model.to(device)
                    model.eval()
                    print(f"Loaded SimpleConvNet model from {args.model_path}")
                except Exception as fallback_error:
                    print(f"Fallback model loading failed: {fallback_error}")
                    result = get_deterministic_result(args.image_path, args.reference_number)
                    result["error"] = f"Model loading failed: {fallback_error}"
                    print(json.dumps(result, indent=2))
                    return
        else:
            print(f"Unsupported model type: {args.model_type}, using mock prediction")
            result = get_deterministic_result(args.image_path, args.reference_number)
            print(json.dumps(result, indent=2))
            return
        
        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image_tensor = preprocess_image(args.image_path, transform)
        
        # Make prediction
        predicted_class, probabilities = predict_image(model, image_tensor, device)
        
        # Process results
        processing_time = time.time() - start_time
        diagnosis = "Pneumonia" if predicted_class == 1 else "Normal"
        confidence = float(probabilities[predicted_class]) * 100
        
        # Create result dictionary
        result = {
            "diagnosis": diagnosis,
            "confidence": round(confidence),
            "processingTime": round(processing_time, 2),
            "probabilities": {
                "normal": round(float(probabilities[0]) * 100),
                "pneumonia": round(float(probabilities[1]) * 100)
            }
        }
        
        # Add pneumonia specific info if positive
        if diagnosis == "Pneumonia":
            # Determine pneumonia type based on confidence
            pneumonia_type = "Bacterial" if confidence > 75 else "Viral"
            # Determine severity based on confidence
            if confidence > 90:
                severity = "Severe"
                severity_desc = "Severe pneumonia with significant lung involvement."
                action = "Immediate medical consultation and treatment recommended."
            elif confidence > 80:
                severity = "Moderate"
                severity_desc = "Moderate pneumonia with partial lung involvement."
                action = "Medical consultation recommended to determine appropriate treatment."
            else:
                severity = "Mild"
                severity_desc = "Mild pneumonia with limited lung involvement."
                action = "Monitor symptoms and consult with a healthcare provider."
                
            result["pneumoniaType"] = pneumonia_type
            result["severity"] = severity
            result["severityDescription"] = severity_desc
            result["recommendedAction"] = action
        else:
            result["recommendedAction"] = "No pneumonia detected. Regular health maintenance recommended."
        
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        # If anything goes wrong, use mock prediction
        print(f"Error: {str(e)}")
        result = get_deterministic_result(args.image_path, args.reference_number)
        result["error"] = str(e)
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
