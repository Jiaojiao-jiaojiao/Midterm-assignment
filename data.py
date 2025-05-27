import os
import torchvision.transforms as transforms  # 修复：添加缺失的导入
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader

def get_caltech101_loaders(root, batch_size=32, val_ratio=0.3):
    # 添加路径检查
    if not os.path.exists(root):
        raise FileNotFoundError(f"数据集路径不存在: {root}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = ImageFolder(root=root, transform=transform)
    
    val_len = int(len(dataset) * val_ratio)
    train_len = len(dataset) - val_len
    train_data, val_data = random_split(dataset, [train_len, val_len])
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader