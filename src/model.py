import torch
from torchvision import models
from torchvision.models import MobileNet_V2_Weights
from torch import nn

def get_model(num_classes):
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model