# Caltech-101分类微调实验

本仓库包含使用ImageNet预训练模型在Caltech-101数据集上进行微调的代码实现。

## 实验环境

- Python 3.8+
- PyTorch 1.8+
- torchvision 0.9+
- CUDA 11.1 (GPU训练需要)

## 数据集准备

1. 下载Caltech-101数据集：
```bash
wget https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip
unzip caltech-101.zip
```
---

## 数据集目录结构

```
caltech-101/
    ├── annotations/
    ├── images/
    └── splits/
```

---
