"""
Fix Python module import issues for the EfficientNet model
"""

import os
import sys
import shutil

def fix_imports():
    """
    Create an __init__.py file in the pneumonia-ml-efficientnet directory 
    to make it importable as a Python package
    """
    # Get the path to the pneumonia-ml-efficientnet directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_dir = os.path.join(project_root, 'pneumonia-ml-efficientnet')
    
    if not os.path.exists(model_dir):
        print(f"Error: Model directory not found at {model_dir}")
        return False
    
    # Create __init__.py file if it doesn't exist
    init_file = os.path.join(model_dir, '__init__.py')
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write("# Make pneumonia-ml-efficientnet importable as a Python package\n")
        print(f"Created {init_file}")
    else:
        print(f"{init_file} already exists")
    
    # If there's a model.py in the directory, make sure it's properly set up
    model_file = os.path.join(model_dir, 'model.py')
    if os.path.exists(model_file):
        print(f"Found model.py at {model_file}")
        
        # Check if EfficientNetModel is defined in model.py
        with open(model_file, 'r') as f:
            content = f.read()
            if 'class EfficientNetModel' in content:
                print("EfficientNetModel class found in model.py")
            else:
                print("WARNING: EfficientNetModel class not found in model.py")
    else:
        print(f"WARNING: model.py not found at {model_file}")
    
    print("\nCreated necessary Python package files to fix import issues.")
    print("Try running the server again with:")
    print("cd backend/ml_model")
    print("python server.py")

if __name__ == "__main__":
    fix_imports() 