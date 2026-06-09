# Classical_Model_-FashionMNIST-

用LeNet、AlexNet、GoogLeNet和ResNet进行FashionMNIST分类任务，同时包含ViT在CIFAR-10上的分类实现。

## 📋 项目概述

本项目实现了多种经典卷积神经网络架构，用于服装图像分类任务。主要包括：

- **LeNet**：经典的卷积神经网络，适合小尺寸图像
- **AlexNet**：深度学习的标志性模型，引入ReLU激活函数
- **GoogLeNet**：引入Inception模块的网络结构
- **ResNet**：残差网络，支持更深层的网络训练
- **ViT**：Vision Transformer，在CIFAR-10数据集上的实现

## 🎯 数据集

- **FashionMNIST**：用于LeNet、AlexNet、GoogLeNet和ResNet
  - 60,000张训练图像
  - 10,000张测试图像
  - 10个服装类别
  - 图像大小：28×28像素

- **CIFAR-10**：用于ViT模型
  - 50,000张训练图像
  - 10,000张测试图像
  - 10个物体类别
  - 图像大小：32×32像素

## 📁 项目结构

```
Classical_Model_-FashionMNIST-/
├── README.md                 # 项目说明文档
├── models/                   # 模型实现
│   ├── lenet.py
│   ├── alexnet.py
│   ├── googlenet.py
│   ├── resnet.py
│   └── vit.py
├── data/                     # 数据处理脚本
│   └── data_loader.py
├── train.py                  # 训练脚本
├── test.py                   # 测试脚本
└── requirements.txt          # 依赖库
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置数据集路径

在运行脚本前，请修改代码中的数据集路径，确保指向正确的数据集目录。

### 3. 训练模型

```bash
# 训练FashionMNIST模型（选择一个）
python train.py --model lenet --dataset fashionmnist
python train.py --model alexnet --dataset fashionmnist
python train.py --model googlenet --dataset fashionmnist
python train.py --model resnet --dataset fashionmnist

# 训练ViT模型（CIFAR-10）
python train.py --model vit --dataset cifar10
```

### 4. 测试模型

```bash
python test.py --model [model_name] --dataset [dataset_name]
```

## 📊 模型性能

| 模型 | 数据集 | 准确率 | 参数量 |
|------|--------|--------|--------|
| LeNet | FashionMNIST | - | ~60K |
| AlexNet | FashionMNIST | - | ~60M |
| GoogLeNet | FashionMNIST | - | ~6.6M |
| ResNet | FashionMNIST | - | ~25M |
| ViT | CIFAR-10 | - | ~86M |

*注：性能数据需要根据实际训练结果填写*

## 🔧 依赖库

- PyTorch >= 1.9
- torchvision >= 0.10
- numpy
- matplotlib
- tqdm

## 📝 详细说明

### FashionMNIST分类

各个模型都可以直接用于FashionMNIST分类任务。FashionMNIST是MNIST的升级版本，包含10种不同的服装类别。

### CIFAR-10分类

ViT模型在CIFAR-10数据集上实现了Vision Transformer架构，将图像分割成Patch进行处理。

## ⚠️ 重要提示

- **修改数据集路径**：在运行脚本前，必须根据本地环境修改数据集的路径配置
- **GPU支持**：建议使用GPU加速训练，可提高训练速度
- **内存需求**：部分模型（如ResNet、ViT）需要较多内存，请确保系统配置充足

## 📖 参考文献

- LeNet: [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
- AlexNet: [ImageNet Classification with Deep CNNs](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- GoogLeNet: [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)
- ResNet: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- ViT: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)

## 📄 许可证

MIT License

## 👤 作者

Fryderyk00cby

## 💬 贡献

欢迎提交Issue和Pull Request！

