import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_model(num_classes, freeze_backbone=True):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
