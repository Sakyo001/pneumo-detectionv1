"""
Debug script to test EfficientNet model loading
"""

import os
import sys
import time

# Add parent directory to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

try:
    # Get the path to the pneumonia-ml-efficientnet directory
    model_dir = os.path.join(project_root, 'pneumonia-ml-efficientnet')
    print(f"Looking for model in: {model_dir}")
    
    # Add model directory to path
    sys.path.append(model_dir)
    
    # Try to import from model.py directly
    print("Trying to import EfficientNetModel...")
    try:
        from model import EfficientNetModel
        print("✅ Successfully imported EfficientNetModel")
    except ImportError:
        print("❌ Failed to import from model.py in current directory")
        # Try with direct path
        model_file = os.path.join(model_dir, 'model.py')
        if os.path.exists(model_file):
            print(f"Found model.py at {model_file}")
            sys.path.insert(0, model_dir)
            try:
                from model import EfficientNetModel
                print("✅ Successfully imported EfficientNetModel with direct path")
            except ImportError as e:
                print(f"❌ Still failed to import: {e}")
        else:
            print(f"❌ model.py not found at {model_file}")
    
    # Check for output directory in project root
    root_output_dir = os.path.join(project_root, 'output')
    print(f"Checking for models in project root output directory: {root_output_dir}")
    root_best_model_path = os.path.join(root_output_dir, 'best_model.pth')
    root_final_model_path = os.path.join(root_output_dir, 'final_model.pth')
    
    print(f"Root output directory exists: {os.path.exists(root_output_dir)}")
    print(f"best_model.pth exists in root output: {os.path.exists(root_best_model_path)}")
    print(f"final_model.pth exists in root output: {os.path.exists(root_final_model_path)}")
    
    if os.path.exists(root_output_dir):
        print("Root output directory contents:")
        for item in os.listdir(root_output_dir):
            file_path = os.path.join(root_output_dir, item)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
            print(f"  - {item} ({file_size:.2f} MB)")
    
    # Try to import torch
    print("\nChecking PyTorch installation...")
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Try to load model from root output directory
    model_path = root_best_model_path if os.path.exists(root_best_model_path) else root_final_model_path
    
    if os.path.exists(model_path):
        print(f"\nAttempting to load model from: {model_path}")
        try:
            model = EfficientNetModel(model_version='b0', pretrained=False)
            # Load the model directly with torch.load first to check if file is valid
            try:
                state_dict = torch.load(model_path, map_location='cpu')
                print(f"✅ Successfully loaded state dict with {len(state_dict)} keys")
                
                # Now try to load into model
                model.load(model_path)
                print("✅ Model loaded successfully!")
                
                # Create a test input
                dummy_input = torch.randn(1, 3, 224, 224)
                try:
                    with torch.no_grad():
                        output = model(dummy_input)
                    print(f"✅ Model inference successful! Output shape: {output.shape}")
                except Exception as inf_err:
                    print(f"❌ Model inference failed: {inf_err}")
                
            except Exception as e:
                print(f"❌ Error loading state dict: {str(e)}")
                import traceback
                traceback.print_exc()
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ No model file found!")
    
except Exception as e:
    print(f"Error in debug script: {str(e)}")
    import traceback
    traceback.print_exc()

print("\nDebugging complete. Check the output above for errors.") 