import torchvision.models as models
from torch import nn

def get_model(num_classes=101, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    for param in model.fc.parameters():
        param.requires_grad = True
        
    return model
