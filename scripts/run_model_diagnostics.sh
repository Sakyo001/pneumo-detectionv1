#!/bin/bash

# Get the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODEL_DIR="$DIR/../output"

echo "========================================="
echo " Model Diagnostics Tool"
echo "========================================="

# Check Python and PyTorch
echo ""
echo "Checking Python and PyTorch..."
python --version
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torchvision; print(f'Torchvision version: {torchvision.__version__}')"

# List model files
echo ""
echo "Checking model files..."
ls -la "$MODEL_DIR"

# Run detailed analysis on final_model.pth
FINAL_MODEL="$MODEL_DIR/final_model.pth"
if [ -f "$FINAL_MODEL" ]; then
    echo ""
    echo "Analyzing final_model.pth..."
    python "$DIR/model_debug.py" --model "$FINAL_MODEL"
fi

# Run detailed analysis on best_model.pth
BEST_MODEL="$MODEL_DIR/best_model.pth"
if [ -f "$BEST_MODEL" ]; then
    echo ""
    echo "Analyzing best_model.pth..."
    python "$DIR/model_debug.py" --model "$BEST_MODEL"
fi

# Test prediction with a sample image
echo ""
echo "Running sample prediction test..."
SAMPLE_DIR="$DIR/../public/samples"
mkdir -p "$SAMPLE_DIR"

# Create a sample image directory if it doesn't exist
if [ ! -d "$SAMPLE_DIR" ]; then
    echo "Sample directory doesn't exist. Please place a test X-ray image in the public/samples directory."
else
    # Find first image in the samples directory
    SAMPLE_IMAGE=$(find "$SAMPLE_DIR" -type f -name "*.jpg" -o -name "*.png" | head -1)
    
    if [ -z "$SAMPLE_IMAGE" ]; then
        echo "No sample images found. Please place a test X-ray image in the public/samples directory."
    else
        echo "Using sample image: $SAMPLE_IMAGE"
        echo ""
        echo "Testing with predict_exact.py..."
        python "$DIR/predict_exact.py" --image "$SAMPLE_IMAGE" --model "$FINAL_MODEL" --fallback
    fi
fi

echo ""
echo "Diagnostics complete." 