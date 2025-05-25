import torch.nn as nn
import torchvision.models as models


def get_model(model_name="resnet18", pretrained=True, num_classes=101):

    if pretrained:

        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:

        model = models.resnet18(weights='DEFAULT')

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    
    return model
