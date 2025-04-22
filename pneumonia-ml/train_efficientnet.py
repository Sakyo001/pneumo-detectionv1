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
from torch.utils.tensorboard import SummaryWriter

# Import our EfficientNet model
from efficientnet_model import EfficientNetModel, get_efficientnet

def train_model(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))
    
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
    model = get_efficientnet(
        variant=args.model_variant, 
        pretrained=args.pretrained, 
        freeze_backbone=args.freeze_backbone
    )
    
    model = model.to(device)
    
    # Log model graph to TensorBoard
    dummy_input = torch.zeros(1, 3, 224, 224).to(device)
    writer.add_graph(model, dummy_input)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Only optimize the parameters with requires_grad=True
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    num_epochs = args.epochs
    best_acc = 0.0
    best_model_wts = None
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    start_time = time.time()
    
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
            
            # Log to TensorBoard
            writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
            writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)
            
            # Record history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                # Adjust learning rate based on validation loss
                scheduler.step(epoch_loss)
                
                # Log learning rate
                writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
            
            # Deep copy the model if it's the best so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict().copy()
                # Save best model
                model_name = f'efficientnet_{args.model_variant}_best.pth'
                torch.save(best_model_wts, os.path.join(args.output_dir, model_name))
                print(f'New best model saved with accuracy: {best_acc:.4f}')
    
    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
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
    
    # Log confusion matrix as image to TensorBoard
    import itertools
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(args.output_dir, f'efficientnet_{args.model_variant}_confusion_matrix.png'))
    
    # Add confusion matrix to TensorBoard
    writer.add_figure('Confusion Matrix', plt.gcf())
    
    # Print classification report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("Classification Report:")
    print(report)
    
    # Save classification report to a file
    with open(os.path.join(args.output_dir, f'efficientnet_{args.model_variant}_classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Save final model
    model_name = f'efficientnet_{args.model_variant}_final.pth'
    torch.save(model.state_dict(), os.path.join(args.output_dir, model_name))
    
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
    plt.savefig(os.path.join(args.output_dir, f'efficientnet_{args.model_variant}_history.png'))
    plt.close()
    
    # Add the figure to TensorBoard
    writer.add_figure('Training History', plt.gcf())
    
    # Close TensorBoard writer
    writer.close()
    
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Pneumonia Detection Model with EfficientNet')
    parser.add_argument('--data_dir', type=str, default='chest_xray', 
                        help='Directory with chest X-ray dataset')
    parser.add_argument('--output_dir', type=str, default='output/efficientnet', 
                        help='Directory to save model checkpoints and results')
    parser.add_argument('--model_variant', type=str, choices=['b0', 'b1', 'b2', 'b3'], default='b0',
                        help='EfficientNet model variant to use (b0, b1, b2, or b3)')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, 
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, 
                        help='Weight decay for optimizer')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of workers for dataloaders')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained model weights')
    parser.add_argument('--no_pretrained', dest='pretrained', action='store_false',
                        help='Do not use pretrained model weights')
    parser.set_defaults(pretrained=True)
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze backbone parameters')
    parser.add_argument('--no_freeze_backbone', dest='freeze_backbone', action='store_false',
                        help='Do not freeze backbone parameters')
    parser.set_defaults(freeze_backbone=True)
    
    args = parser.parse_args()
    train_model(args) 