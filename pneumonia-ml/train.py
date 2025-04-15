import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import PneumoniaModel, SimpleConvNet

def train_model(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data augmentation and normalization for training
    # Just normalization for validation and testing
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # Create dataset
    data_dir = args.data_dir
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val', 'test']
    }
    
    # Get class names
    class_names = image_datasets['train'].classes
    print(f"Classes: {class_names}")
    
    # Create dataloaders
    dataloaders = {
        x: DataLoader(
            image_datasets[x], 
            batch_size=args.batch_size,
            shuffle=True if x == 'train' else False,
            num_workers=args.num_workers
        )
        for x in ['train', 'val', 'test']
    }
    
    # Calculate class weights to address imbalance
    train_counts = [0, 0]
    for _, label in image_datasets['train'].samples:
        train_counts[label] += 1
    total = sum(train_counts)
    class_weights = torch.FloatTensor([total/count for count in train_counts]).to(device)
    print(f"Class weights: {class_weights}")
    
    # Initialize model
    if args.model_type == 'resnet':
        model = PneumoniaModel(pretrained=True, freeze_backbone=args.freeze_backbone)
    else:
        model = SimpleConvNet()
    
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Only optimize the unfrozen parameters
    if args.model_type == 'resnet' and args.freeze_backbone:
        optimizer = optim.Adam(model.backbone.fc.parameters(), lr=args.learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    num_epochs = args.epochs
    best_acc = 0.0
    best_model_wts = None
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            pbar = tqdm(dataloaders[phase], desc=f'{phase.capitalize()} Batch')
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass - only track history in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': torch.sum(preds == labels.data).item() / inputs.size(0)
                })
            
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Record history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                # Adjust learning rate based on validation loss
                scheduler.step(epoch_loss)
            
            # Deep copy the model if it's the best so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict().copy()
                # Save best model
                torch.save(best_model_wts, os.path.join(args.output_dir, 'best_model.pth'))
                print(f'New best model saved with accuracy: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Final model evaluation on test set
    model.eval()
    test_running_corrects = 0
    test_running_total = 0
    
    # Track predictions and true labels for confusion matrix
    all_preds = []
    all_labels = []
    
    print("Evaluating on test set...")
    with torch.no_grad():
        for inputs, labels in tqdm(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            test_running_corrects += torch.sum(preds == labels.data)
            test_running_total += inputs.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = test_running_corrects.double() / test_running_total
    print(f'Test Accuracy: {test_acc:.4f}')
    
    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_history.png'))
    plt.close()
    
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Pneumonia Detection Model')
    parser.add_argument('--data_dir', type=str, default='chest_xray', 
                        help='Directory with chest X-ray dataset')
    parser.add_argument('--output_dir', type=str, default='output', 
                        help='Directory to save model checkpoints and results')
    parser.add_argument('--model_type', type=str, choices=['resnet', 'simple'], default='resnet',
                        help='Model architecture to use (resnet or simple CNN)')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of worker threads for data loading')
    parser.add_argument('--freeze_backbone', type=bool, default=True, 
                        help='Whether to freeze backbone network for transfer learning')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train model
    model = train_model(args)
    
    print("Training completed!")
