import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

class Residual(nn.Module):
    def __init__(self,input_channel,output_channel,use_1=False,strides=1):
        super().__init__()
        self.ReLU = nn.ReLU()
        self.conv1 = nn.Conv2d(input_channel,output_channel,kernel_size=3,padding=1,stride=strides)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel,output_channel,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(output_channel)
        if use_1:
            self.conv3 = nn.Conv2d(input_channel,output_channel,kernel_size=1,stride=strides)
        else:
            self.conv3 = None
    def forward(self,x):
        out = self.ReLU(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.conv3:
            x = self.conv3(x)
        return self.ReLU(out+x)

class ResNet18(nn.Module):
    def __init__(self,Residual):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=7,stride=2,padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.block2 = nn.Sequential(
            Residual(64,64),
            Residual(64,64)
        )
        self.block3 = nn.Sequential(
            Residual(64,128,use_1=True,strides=2),
            Residual(128,128)
        )
        self.block4 = nn.Sequential(
            Residual(128,256,use_1=True,strides=2),
            Residual(256,256)
        )
        self.block5 = nn.Sequential(
            Residual(256,512,use_1=True,strides=2),
            Residual(512,512)
        )
        self.block6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512,10)
        )
        #kaiming初始化
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.constant_(m.bias,0)

    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet18(Residual).to(device)
    print(summary(model,(1,224,224)))
        