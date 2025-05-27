from data import get_caltech101_loaders
from models import get_model
from train import train_model
import os

# 修复：检查数据集路径
data_root = '101_ObjectCategories'
if not os.path.exists(data_root):
    raise FileNotFoundError(f"请将数据集文件夹'101_ObjectCategories'放在与run.py同级目录下")

train_loader, val_loader = get_caltech101_loaders(data_root)

# 实验1：预训练模型
model1 = get_model(pretrained=True)
acc1 = train_model(model1, train_loader, val_loader, save_path="results/model_pretrained.pt")
print(f"预训练模型验证准确率: {acc1:.2%}")

# 实验2：随机初始化
model2 = get_model(pretrained=False)
acc2 = train_model(model2, train_loader, val_loader, save_path="results/model_random.pt")
print(f"随机初始化模型验证准确率: {acc2:.2%}")