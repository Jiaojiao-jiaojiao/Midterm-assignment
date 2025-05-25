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


# 基于VOC数据集的Mask R-CNN与Sparse R-CNN目标检测实验

本项目使用mmdetection框架在PASCAL VOC数据集上实现了Mask R-CNN和Sparse R-CNN模型的训练与测试，对比分析了两种模型在目标检测和实例分割任务上的性能差异。

---

## 项目简介

- **任务类型**：目标检测 / 实例分割
- **基础框架**：[mmdetection v2.25.0](https://github.com/open-mmlab/mmdetection)
- **对比模型**：
  - Mask R-CNN（两阶段检测器）
  - Sparse R-CNN（端到端稀疏检测器）
- **核心实验**：
  - Proposal box与最终预测结果可视化对比
  - 跨数据集泛化能力测试

---

## 项目文件结构
```bash
.
├── configs/                          # 模型配置文件
│   ├── mask_rcnn/                    # Mask R-CNN配置
│   │   ├── mask_rcnn_r50_fpn_voc.py  # 模型架构
│   │   └── schedule_voc.py           # 训练策略
│   └── sparse_rcnn/                  # Sparse R-CNN配置
├── data/                             # 数据管理
│   ├── voc0712/                      # VOC数据集
│   └── custom_images/                # 自定义测试图像
├── tools/                            # 实用工具
│   ├── train.py                      # 训练脚本
│   ├── test.py                       # 测试脚本
│   └── visualization/                # 可视化工具
├── outputs/                          # 实验输出
│   ├── checkpoints/                  # 模型权重
│   ├── logs/                         # TensorBoard日志
│   └── predictions/                  # 预测结果可视化
└── README.md                         # 项目说明文档
```

---

## 实验配置

### 关键参数设置
| 参数项          | Mask R-CNN          | Sparse R-CNN        |
|----------------|---------------------|---------------------|
| Backbone       | ResNet-50-FPN       | ResNet-50-FPN       |
| Batch Size     | 8                   | 8                   |
| Base LR        | 0.02                | 0.01                |
| Optimizer      | SGD                 | AdamW               |
| Epochs         | 24                  | 36                  |
| Schedule       | StepLR (16,22)      | CosineAnnealing     |
| Proposal Num   | 1000                | 300 (learnable)     |

---

## 快速开始

### 1. 环境安装
```bash
conda create -n mmdet python=3.8 -y
conda activate mmdet
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.5 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection && pip install -v -e .
```

### 2. 数据准备
```bash
# 下载VOC数据集
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar -C data/voc0712/

# 转换为COCO格式
python tools/dataset_converters/pascal_voc.py data/voc0712 --out-dir data/voc0712_coco
```

### 3. 模型训练
```bash
# Mask R-CNN训练
python tools/train.py configs/mask_rcnn/mask_rcnn_r50_fpn_voc.py --work-dir outputs/mask_rcnn

# Sparse R-CNN训练
python tools/train.py configs/sparse_rcnn/sparse_rcnn_r50_fpn_voc.py --work-dir outputs/sparse_rcnn
```

### 4. 结果可视化
```python
from mmdet.apis import init_detector, inference_detector, show_result_pyplot

# 加载模型
model = init_detector('configs/mask_rcnn/mask_rcnn_r50_fpn_voc.py', 'outputs/mask_rcnn/latest.pth')

# 可视化预测
result = inference_detector(model, 'data/custom_images/demo.jpg')
show_result_pyplot(model, 'data/custom_images/demo.jpg', result, score_thr=0.5)
```

---

## 实验结果

### 性能指标 (VOC test)
| 模型          | mAP@0.5 | mAP@0.5:0.95 | 推理速度(FPS) |
|---------------|---------|--------------|--------------|
| Mask R-CNN    | 78.4    | 56.2         | 12.3         |
| Sparse R-CNN  | 75.8    | 53.7         | 18.6         |

### 可视化对比
![](https://via.placeholder.com/600x300?text=Mask+R-CNN+vs+Sparse+R-CNN+Prediction+Comparison)

---

## 模型下载

- **预训练权重**：
  - Google Drive: [Mask R-CNN](https://drive.google.com/xxx) | [Sparse R-CNN](https://drive.google.com/xxx)
  - 百度网盘: [链接](https://pan.baidu.com/xxx) 提取码：`vocmm`

- **训练日志**：
  - [TensorBoard日志](outputs/logs) 包含完整训练曲线

---


   
    


































