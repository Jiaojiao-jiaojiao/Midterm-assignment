from data import get_caltech101_loaders
from models import get_model
from train import train_model
import itertools
import os

# 超参数组合
hyperparams = {
    "lr": [1e-3, 1e-4, 1e-5],
    "epochs": [10, 20],
    "pretrained": [True, False]
}

# 生成所有组合
param_combinations = list(itertools.product(
    hyperparams["lr"], 
    hyperparams["epochs"], 
    hyperparams["pretrained"]
))

# 检查数据集路径
data_root = '101_ObjectCategories'
if not os.path.exists(data_root):
    raise FileNotFoundError(
        f"数据集文件夹'{data_root}'未找到。请确保:\n"
        "1. 已下载Caltech101数据集\n"
        "2. 将文件夹重命名为'101_ObjectCategories'\n"
        "3. 放在与run.py相同的目录下"
    )

# 获取数据加载器
train_loader, val_loader = get_caltech101_loaders(data_root)

# 创建结果目录
os.makedirs("results", exist_ok=True)

# 运行实验
results = []
for lr, epochs, pretrained in param_combinations:
    print(f"\n训练配置: lr={lr}, epochs={epochs}, pretrained={pretrained}")
    
    # 获取模型
    model = get_model(pretrained=pretrained)
    
    # 训练模型
    acc = train_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=epochs, 
        lr=lr, 
        pretrained=pretrained,
        save_dir=f"results/lr_{lr}_epochs_{epochs}_pretrained_{pretrained}"
    )
    
    results.append((lr, epochs, pretrained, acc))

# 打印结果摘要
print("\n=== 结果汇总 ===")
print("学习率\t\t训练轮次\t预训练\t验证准确率")
print("-" * 50)
for r in results:
    print(f"{r[0]:.0e}\t\t{r[1]}\t\t{r[2]}\t{r[3]:.2%}")

# 保存结果到文件
with open("results/summary.txt", "w") as f:
    f.write("学习率,训练轮次,预训练,验证准确率\n")
    for r in results:
        f.write(f"{r[0]},{r[1]},{r[2]},{r[3]}\n")
    print("\n结果已保存到 results/summary.txt")

print("\n实验完成！结果已保存到 results/ 目录")
