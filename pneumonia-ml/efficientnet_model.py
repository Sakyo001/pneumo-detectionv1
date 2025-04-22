import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3,
    EfficientNet_B0_Weights, EfficientNet_B1_Weights, 
    EfficientNet_B2_Weights, EfficientNet_B3_Weights
)

class EfficientNetModel(nn.Module):
    """
    EfficientNet model for pneumonia detection from chest X-rays.
    Supports multiple variants of EfficientNet (B0, B1, B2, B3).
    """
    def __init__(self, model_variant='b0', pretrained=True, freeze_backbone=True, num_classes=2):
        super(EfficientNetModel, self).__init__()
        
        # Initialize EfficientNet model based on variant
        if model_variant == 'b0':
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = efficientnet_b0(weights=weights)
            num_features = self.backbone.classifier[1].in_features
        elif model_variant == 'b1':
            weights = EfficientNet_B1_Weights.DEFAULT if pretrained else None
            self.backbone = efficientnet_b1(weights=weights)
            num_features = self.backbone.classifier[1].in_features
        elif model_variant == 'b2':
            weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
            self.backbone = efficientnet_b2(weights=weights)
            num_features = self.backbone.classifier[1].in_features
        elif model_variant == 'b3':
            weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
            self.backbone = efficientnet_b3(weights=weights)
            num_features = self.backbone.classifier[1].in_features
        else:
            raise ValueError(f"Unsupported model variant: {model_variant}. Choose from b0, b1, b2, b3.")
        
        # Freeze backbone parameters if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace the final fully connected layer for classification
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))
        
    def get_features(self, x):
        """Extract features before final classification layer"""
        # Get all layers except the last one
        feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        return feature_extractor(x)

def get_efficientnet(variant='b0', pretrained=True, freeze_backbone=True, num_classes=2):
    """Helper function to create and initialize an EfficientNet model"""
    return EfficientNetModel(
        model_variant=variant,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        num_classes=num_classes
    ) 