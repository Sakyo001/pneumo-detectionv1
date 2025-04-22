# Pneumonia Detection ML Models

This directory contains machine learning models for pneumonia detection from chest X-ray images.

## Available Models

1. **ResNet50** (`train.py`) - A pretrained ResNet50 model for pneumonia classification
2. **EfficientNet** (`train_efficientnet.py`) - Various EfficientNet models (B0, B1, B2, B3) for pneumonia classification

## Environment Setup

Create a Python virtual environment and install the required dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset Structure

The code expects a dataset with the following structure:

```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

You can download the Chest X-Ray dataset from Kaggle: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

## Training Models

### ResNet50 Model

To train the ResNet50 model:

```bash
python train.py --data_dir chest_xray --output_dir output/resnet --batch_size 16 --epochs 15
```

### EfficientNet Model

To train an EfficientNet model:

```bash
python train_efficientnet.py --data_dir chest_xray --output_dir output/efficientnet --model_variant b0 --batch_size 16 --epochs 15
```

Available model variants are `b0`, `b1`, `b2`, and `b3`. Larger variants have more parameters and may require more computational resources.

## Training Options

Both training scripts support these common options:

- `--data_dir`: Directory containing the chest X-ray dataset
- `--output_dir`: Directory to save model checkpoints and results
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of epochs to train (default: 10 for ResNet, 20 for EfficientNet)
- `--learning_rate`: Initial learning rate (default: 0.001)
- `--num_workers`: Number of workers for data loading (default: 4)
- `--pretrained`: Use pretrained weights (default: True)
- `--no_pretrained`: Do not use pretrained weights
- `--freeze_backbone`: Freeze backbone parameters (default: True)
- `--no_freeze_backbone`: Do not freeze backbone parameters

## Inference

After training, you can perform inference using the `inference.py` script:

```bash
python inference.py --model_path output/efficientnet/efficientnet_b0_best.pth --input_dir test_images --output_dir inference_output --model_type efficientnet --variant b0
```

## Model Results

During training, the following files will be generated in the output directory:

- Model checkpoints (`.pth` files)
- Training history graph
- Confusion matrix
- Classification report
- TensorBoard logs

## TensorBoard Visualization

You can visualize training progress using TensorBoard:

```bash
tensorboard --logdir output/efficientnet/tensorboard
```

## Model Performance

EfficientNet generally offers a good balance of accuracy and efficiency:

| Model | Parameters | Speed | Accuracy* |
|-------|------------|-------|-----------|
| ResNet50 | 25.6M | Medium | Good |
| EfficientNet-B0 | 5.3M | Fast | Good |
| EfficientNet-B3 | 12M | Medium | Better |

*Actual accuracy will depend on your dataset and training parameters. 