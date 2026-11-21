import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights

# Builds a model with an option to freeze the backbone and customize the number of output classes
def build_model(num_classes, arch="resnet18", freeze_backbone=True):
    if arch == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    elif arch == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        raise ValueError(f"Unsupported model: {arch}")

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
