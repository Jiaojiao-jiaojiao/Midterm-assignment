import torch
from torch import nn, optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os  

def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-3, save_path="results/saved_model.pt"):
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 自动创建results目录
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 确保所有参数可训练
    for param in model.parameters():
        param.requires_grad = True
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 创建TensorBoard日志目录
    log_dir = os.path.join(os.path.dirname(save_path), "logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    train_accs, val_accs, train_losses, val_losses = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        correct, total, train_loss = 0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # 记录训练指标
        train_acc = correct / total
        train_loss_avg = train_loss / len(train_loader)
        train_accs.append(train_acc)
        train_losses.append(train_loss_avg)
        writer.add_scalar("Loss/train", train_loss_avg, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)

        # 验证阶段
        model.eval()
        correct, total, val_loss = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        val_loss_avg = val_loss / len(val_loader)
        val_accs.append(val_acc)
        val_losses.append(val_loss_avg)
        writer.add_scalar("Loss/val", val_loss_avg, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss_avg:.4f} | "
              f"Val Accuracy: {val_acc:.2%}")

    # 保存模型和关闭TensorBoard
    torch.save(model.state_dict(), save_path)
    writer.close()
    return val_acc