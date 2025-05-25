# Caltech-101分类微调实验

本项目通过微调在ImageNet上预训练的卷积神经网络（如AlexNet或ResNet-18），实现对Caltech-101数据集的图像分类任务

---

## 项目简介

- **基础模型**：ImageNet预训练的AlexNet/ResNet-18

- **修改部分**：替换输出层为101类（适配Caltech-101类别数）

- **训练策略**：

  - 冻结预训练层，仅训练新输出层

  - 使用较小学习率微调全部网络参数

- **对比实验**：与从零开始训练的模型进行性能对比

---

## 项目文件结构

```bash

├── models/                # 网络模型定义
│   ├── alexnet.py         # AlexNet实现
│   └── resnet.py          # ResNet-18实现
├── data/                  # 数据加载与预处理
│   ├── caltech101.py      # Caltech-101数据集加载
│   └── transforms.py      # 数据增强
├── train.py               # 模型训练脚本
├── test.py                # 模型测试脚本
├── utils/                 # 辅助工具
│   ├── logger.py          # 训练日志记录
│   └── visualize.py       # TensorBoard可视化
├── configs/               # 配置文件
│   ├── base.yaml          # 基础配置
│   └── experiments/       # 实验配置
├── outputs/               # 训练输出
│   ├── checkpoints/       # 模型权重
│   └── logs/              # 训练日志
└── README.md              # 项目说明文档

```

---


## 模型结构

### 微调策略

1、**基础架构**：保留预训练CNN的全部卷积层

2、**输出层**：替换原始分类器为：

   - 全连接层（AlexNet）

   - 1x1卷积+全局池化（ResNet）

3、**参数初始化**：

   - 预训练参数作为初始值

   - 新分类层随机初始化

---

## 训练配置

### 超参数设置

支持以下关键参数配置：

```yaml

# 学习策略
learning_rate: 1e-3        # 基础学习率
finetune_lr: 1e-5         # 微调学习率
batch_size: 32
epochs: 50

# 数据增强
augmentation:
  random_crop: True
  horizontal_flip: True
  color_jitter: 0.1

# 优化器
optimizer: adam
weight_decay: 1e-4

```

---

### 训练模式

```bash

# 仅训练新分类层
python train.py --phase feature_extract

# 全网络微调
python train.py --phase finetune

```

---

## 训练过程

### 两阶段训练

1、**特征提取阶段**：
   
   - 冻结预训练层参数

   - 仅训练新添加的分类层

   - 使用较大学习率（1e-3）

2、**微调阶段**：

   - 解冻全部网络层

   - 使用较小学习率（1e-5）微调

   - 启用数据增强

### 监控指标

- 训练/验证损失曲线

- Top-1/Top-5准确率

- 参数梯度分布

---

## 实验结果

### 性能对比

| 方法               | Top-1 Acc  | Top-5 Acc | 
|--------------------|--------|------------|
| 从零训练 | 58.2%  | 11.2       | 
| 预训练+微调 | 76.8% | 11.2       | 

### 可视化结果

- TensorBoard训练曲线

- 混淆矩阵分析

- 特征空间可视化（t-SNE）

---

## 模型下载

  - **预训练权重**：
    
    - Google Drive: 链接

    - 提取码：caltech
   
  - **微调后模型**：
 
    - 百度网盘: 链接

    - 提取码：101finetune
   
---

## 后续改进

- 尝试不同预训练模型（VGG, EfficientNet）

- 引入注意力机制

- 使用更复杂的数据增强策略

- 实现模型剪枝/量化


---
   
    


































