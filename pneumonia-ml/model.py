import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class PneumoniaModel(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=True):
        super(PneumoniaModel, self).__init__()
        # Use a pretrained ResNet50 as backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Freeze backbone parameters to prevent overfitting
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
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
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))


class SimpleConvNet(nn.Module):
    """A simpler CNN model if you have limited computational resources"""
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # Input is expected to be 224x224
        x = self.pool(F.relu(self.conv1(x)))  # 112x112
        x = self.pool(F.relu(self.conv2(x)))  # 56x56
        x = self.pool(F.relu(self.conv3(x)))  # 28x28
        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))
