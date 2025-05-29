# Caltech-101分类微调实验

本项目通过微调在ImageNet上预训练的卷积神经网络ResNet-18，实现对Caltech-101数据集的图像分类任务

---

## 项目简介

- **基础模型**：ResNet-18

- **修改部分**：替换输出层为101类（适配Caltech-101类别数）

- **对比实验**：与从零开始训练的模型进行性能对比

---

## 项目文件结构

```bash

├── models.py               # 网络模型定义
├── data.py                 #  Caltech-101数据集加载与预处理
├── train.py               # 模型训练脚本
├── run.py                 # 训练输出
└── README.md              # 项目说明文档

```

---


## 模型结构

模型采用的是ResNet-18变体：

 - 输入尺寸：3×224×224（RGB图像）

 - 输出维度：101（对应Caltech-101类别数）


## 预训练模式：

 - 使用ImageNet1K_V1预训练权重初始化

 - 仅最后一层全连接层随机初始化

## 从零训练模式：

 - 所有权重随机初始化

---

## 实验设置

### 训练设置

 - 优化器：Adam（β1=0.9, β2=0.999）

 - 损失函数：交叉熵损失

 - 批量大小：32

 - 训练设备：优先使用CUDA GPU

 - 早停机制：保留验证集最佳模型

### 超参数搜索空间

| 参数  | 取值 | 
|--------|------------|
|学习率 |	1e-2, 1e-3, 1e-4 |
| 训练轮次 |	5, 10      | 
|是否预训练 |	是, 否 |

   
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
│   │   └── mask_rcnn_r50_fpn_voc.py           
│   └── sparse_rcnn/                  # Sparse R-CNN配置
        └── mask_rcnn_r50_fpn_voc.py                  
├── data/                             # 数据管理
│   └── voc0712/                      # VOC数据集
├── tools/                            # 实用工具
│   ├── train.py                      # 训练脚本
│   └── infer_and_visualize.py         # 可视化工具
└── README.md                         # 项目说明文档

```

---

## 实验配置

### 模型配置对比

| 参数项          | Mask R-CNN          | Sparse R-CNN        |
|----------------|---------------------|---------------------|
| 基础配置文件    | mask\_rcnn\_r50\_fpn\_1x\_coco.py |sparse\_rcnn\_r50\_fpn\_1x\_coco.py       |
| 输入尺寸     | 1000×600                   | 1000×600                  |
| 批大小       | 2                | 2                |
| 数据增强      | 随机水平翻转(0.5)    | 随机水平翻转(0.5)              |
| 训练轮次        | 12                  | 12                 |

### 关键参数

 - 验证间隔：每1epoch验证(val\_interval=1)
 - 日志记录：TensorboardLoggerHook



---

## 后续改进

 - 添加验证指标：VOCMetric (mAP)
 - 对比不同输入尺寸的影响
 - 测试学习率策略的效果
 - 实现交叉验证
 - 增加学习率搜索功能



---


   
    


































