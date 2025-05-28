import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torch.nn as nn

def get_model(pretrained=True):
    """获取模型，使用新版weights参数替代pretrained"""
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    
    # 修改最后一层全连接层（适配Caltech101的101个类别）
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 101)
    
    return model

