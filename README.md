# Classical_Model_-FashionMNIST-

Implementation of LeNet, AlexNet, GoogLeNet, and ResNet for FashionMNIST classification tasks, along with Vision Transformer (ViT) implementation on CIFAR-10.

## 📋 Project Overview

This project implements multiple classical convolutional neural network architectures for clothing image classification. Main components include:

- **LeNet**: Classic convolutional neural network, suitable for small-sized images
- **AlexNet**: Landmark deep learning model, introducing ReLU activation functions
- **GoogLeNet**: Network architecture introducing Inception modules
- **ResNet**: Residual network, supporting deeper network training
- **ViT**: Vision Transformer, implemented on CIFAR-10 dataset

## 🎯 Datasets

- **FashionMNIST**: Used for LeNet, AlexNet, GoogLeNet, and ResNet
  - 60,000 training images
  - 10,000 test images
  - 10 clothing categories
  - Image size: 28×28 pixels

- **CIFAR-10**: Used for ViT model
  - 50,000 training images
  - 10,000 test images
  - 10 object categories
  - Image size: 32×32 pixels

## 📁 Project Structure

```
Classical_Model_-FashionMNIST-/
├── README.md                 # Project documentation
├── models/                   # Model implementations
│   ├── lenet.py
│   ├── alexnet.py
│   ├── googlenet.py
│   ├── resnet.py
│   └── vit.py
├── data/                     # Data processing scripts
│   └── data_loader.py
├── train.py                  # Training script
├── test.py                   # Testing script
└── requirements.txt          # Dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Dataset Path

Before running the scripts, please modify the dataset path in the code to ensure it points to the correct dataset directory.

### 3. Train Models

```bash
# Train FashionMNIST models (choose one)
python train.py --model lenet --dataset fashionmnist
python train.py --model alexnet --dataset fashionmnist
python train.py --model googlenet --dataset fashionmnist
python train.py --model resnet --dataset fashionmnist

# Train ViT model (CIFAR-10)
python train.py --model vit --dataset cifar10
```

### 4. Test Models

```bash
python test.py --model [model_name] --dataset [dataset_name]
```

## 📊 Model Performance

| Model | Dataset | Parameters |
|-------|---------|-----------|
| LeNet | FashionMNIST | ~60K |
| AlexNet | FashionMNIST | ~60M |
| GoogLeNet | FashionMNIST | ~6.6M |
| ResNet | FashionMNIST | ~25M |
| ViT | CIFAR-10 | ~86M |

## 🔧 Dependencies

- PyTorch >= 1.9
- torchvision >= 0.10
- numpy
- matplotlib
- tqdm

## 📝 Detailed Description

### FashionMNIST Classification

All models can be directly used for FashionMNIST classification tasks. FashionMNIST is an upgraded version of MNIST, containing 10 different clothing categories.

### CIFAR-10 Classification

The ViT model implements Vision Transformer architecture on CIFAR-10 dataset, processing images by splitting them into patches.

## ⚠️ Important Notes

- **Dataset Path Configuration**: Before running scripts, you must modify the dataset path configuration according to your local environment
- **GPU Support**: GPU acceleration is recommended for faster training
- **Memory Requirements**: Some models (such as ResNet and ViT) require significant memory. Please ensure your system has sufficient resources

## 📖 References

- LeNet: [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
- AlexNet: [ImageNet Classification with Deep CNNs](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- GoogLeNet: [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)
- ResNet: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- ViT: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
