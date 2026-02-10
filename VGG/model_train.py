from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import copy
import time
import pandas as pd

from model import VGG16

def train_val_data_process():
    train_data = FashionMNIST(root = 'D:/code/MyModel/VGG-16/data',
                          train = True,
                          transform=transforms.Compose([transforms.Resize(size=224),transforms.ToTensor()]),
                            download=True)
    train_data,val_data = Data.random_split(train_data,[round(0.8*len(train_data)),round(0.2*len(train_data))])
    train_dataloader = Data.DataLoader(dataset=train_data,
                                        batch_size=32,
                                        shuffle =True,
                                        num_workers=0)
    val_dataloader = Data.DataLoader(dataset=val_data,
                                        batch_size=32,
                                        shuffle =True,
                                        num_workers=0)
    return train_dataloader,val_dataloader

def train_model_process(model,train_dataloader,val_dataloader,num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #adam优化 以及 交叉熵损失函数
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    #复制当前模型参数
    best_model_wts = copy.deepcopy(model.state_dict())
    #初始化参数
    #最高准确度
    best_acc = 0.0
    #训练集和验证集损失函数列表
    train_loss_all = []
    val_loss_all = []
    #train和val的acc
    train_acc_all = []
    val_acc_all = []
    #保存时间
    since  = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch{epoch+1}/{num_epochs}")
        print('-'*10)
        # 单次中参数初始化
        train_loss = 0.0
        train_correct = 0
        val_loss = 0.0
        val_correct = 0
        #验证集和训练集样本数量
        train_num = 0
        val_num = 0
        #训练
        for step,(b_x,b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device) 
            b_y = b_y.to(device) 
            model.train()

            output = model(b_x)
            pre_lab = torch.argmax(output,dim = 1)
            #交叉熵自动softmax
            loss = criterion(output,b_y)
            #将梯度初始化为0
            optimizer.zero_grad()
            #反向传播+参数更新
            loss.backward()
            optimizer.step()

            train_loss+=loss.item()*b_x.size(0)
            train_correct+=torch.sum(pre_lab == b_y.data)
            train_num += b_x.size(0)

        #验证
        with torch.no_grad():
            for step,(b_x,b_y) in enumerate(val_dataloader):
                b_x = b_x.to(device) 
                b_y = b_y.to(device) 
                model.eval()

                output = model(b_x)
                pre_lab = torch.argmax(output,dim = 1)
                loss = criterion(output,b_y)

                val_loss += loss.item()*b_x.size(0)
                val_correct += torch.sum(pre_lab == b_y.data)
                val_num += b_x.size(0)

        train_loss_all.append(train_loss/train_num)
        val_loss_all.append(val_loss/val_num)
        train_acc_all.append(train_correct.double().item()/train_num)
        val_acc_all.append(val_correct.double().item()/val_num)
        print(f'{epoch+1} Train loss :{train_loss_all[-1]:.4f},Train acc:{train_acc_all[-1]:.4f}')
        print(f'{epoch+1} val loss :{val_loss_all[-1]:.4f},val acc:{val_acc_all[-1]:.4f}')
        
        if val_acc_all[-1]>best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        #训练时间
        time_use = time.time()-since
        print(f"训练耗时：{time_use//60:.0f}m{time_use%60:.0f}s")

    #选择最优参数
    #加载最高准确率下的模型参数
    torch.save(best_model_wts, 'best_model.pth') 

    train_process = pd.DataFrame(data={'epoch':range(1,num_epochs+1),
                                        'train_loss':train_loss_all,
                                        'val_loss':val_loss_all,
                                        'train_acc':train_acc_all,
                                        'val_acc':val_acc_all})
    return train_process


def matplot_acc_loss(train_process):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_process['epoch'],train_process['train_loss'],label='train_loss')
    plt.plot(train_process['epoch'],train_process['val_loss'],label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss curve')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(train_process['epoch'],train_process['train_acc'],label='train_acc')
    plt.plot(train_process['epoch'],train_process['val_acc'],label='val_acc')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title('acc curve')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_dataloader,val_dataloader = train_val_data_process()
    model = VGG16()
    train_process = train_model_process(model,train_dataloader,val_dataloader,num_epochs=20)
    matplot_acc_loss(train_process)