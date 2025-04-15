#!/usr/bin/env python

import argparse
import os
import sys

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Error: PyTorch not installed")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description='Get model information')
    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        return
    
    try:
        # Load the model file
        model_data = torch.load(args.model, map_location=torch.device('cpu'))
        
        # Check if it's a state dict or a full model
        if isinstance(model_data, dict):
            print("Model type: State Dictionary (OrderedDict)")
            
            # Count parameters
            total_params = sum([p.numel() for p in model_data.values()])
            
            # Get some keys to identify model type
            keys = list(model_data.keys())
            first_5_keys = keys[:5] if len(keys) >= 5 else keys
            
            # Look for common architecture patterns in the keys
            architecture = "Unknown"
            if any("resnet" in k.lower() for k in keys):
                architecture = "ResNet"
            elif any("efficientnet" in k.lower() for k in keys):
                architecture = "EfficientNet"
            elif any("densenet" in k.lower() for k in keys):
                architecture = "DenseNet"
            elif any("vgg" in k.lower() for k in keys):
                architecture = "VGG"
            elif any("mobilenet" in k.lower() for k in keys):
                architecture = "MobileNet"
            
            # Print summary
            print(f"Architecture: {architecture}")
            print(f"Parameters: {total_params:,}")
            print(f"Sample keys: {', '.join(first_5_keys)}")
            
        elif hasattr(model_data, 'state_dict'):
            # It's a model object
            print("Model type: Full Model")
            
            # Get model class name
            model_class = model_data.__class__.__name__
            
            # Get parameters
            total_params = sum(p.numel() for p in model_data.parameters())
            
            # Print summary
            print(f"Architecture: {model_class}")
            print(f"Parameters: {total_params:,}")
            
            # Try to get more detailed info
            if hasattr(model_data, 'name'):
                print(f"Model name: {model_data.name}")
            
            # Check for common model architectures
            if hasattr(model_data, 'backbone'):
                print(f"Backbone: {model_data.backbone.__class__.__name__}")
                
        else:
            print(f"Unknown model format: {type(model_data)}")
    
    except Exception as e:
        print(f"Error analyzing model: {e}")

if __name__ == "__main__":
    main() 