#!/usr/bin/env python

import argparse
import os
import sys
import torch
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Debug model information')
    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    return parser.parse_args()

def print_state_dict_keys(state_dict):
    """Print the key structure of a state dictionary"""
    print("\nModel State Dictionary Keys:")
    print("=========================")
    
    key_groups = {}
    for key in state_dict.keys():
        parts = key.split('.')
        prefix = parts[0]
        if prefix not in key_groups:
            key_groups[prefix] = []
        key_groups[prefix].append(key)
    
    for prefix, keys in key_groups.items():
        print(f"\n{prefix}:")
        for key in sorted(keys):
            shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
            print(f"  {key}: {shape}")
    
    # Print JSON structure for easy parsing
    print("\nJSON Structure:")
    structure = {}
    for key, value in state_dict.items():
        if hasattr(value, 'shape'):
            structure[key] = {
                'shape': list(value.shape),
                'dtype': str(value.dtype)
            }
        else:
            structure[key] = {
                'type': str(type(value))
            }
    
    print(json.dumps(structure, indent=2))

def main():
    args = parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        return
    
    print(f"Loading model from: {args.model}")
    print(f"PyTorch version: {torch.__version__}")
    
    try:
        # Load the model file
        model_data = torch.load(args.model, map_location=torch.device('cpu'))
        
        # Basic information
        print(f"\nModel data type: {type(model_data)}")
        
        # Handle different model types
        if isinstance(model_data, dict):
            print("\nLoaded as state dictionary (OrderedDict)")
            
            # Get basic stats
            param_count = sum(p.numel() for p in model_data.values() if hasattr(p, 'numel'))
            print(f"Total parameters: {param_count:,}")
            
            # Print key structure
            print_state_dict_keys(model_data)
            
        elif hasattr(model_data, 'state_dict'):
            print("\nLoaded as model object")
            
            # Print model info
            print(f"Model class: {model_data.__class__.__name__}")
            
            # Get state dict
            state_dict = model_data.state_dict()
            param_count = sum(p.numel() for p in state_dict.values() if hasattr(p, 'numel'))
            print(f"Total parameters: {param_count:,}")
            
            # Print architecture summary
            print("\nModel Architecture:")
            print(model_data)
            
            # Print state dict structure
            print_state_dict_keys(state_dict)
        
        else:
            print(f"Unknown model format: {type(model_data)}")
            print(f"Model data attributes: {dir(model_data)}")
    
    except Exception as e:
        print(f"Error analyzing model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 