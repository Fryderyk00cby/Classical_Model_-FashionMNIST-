import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F
#注意 原本AlexNet的输入是3通道的RGB图像，这里我们将其修改为1通道的灰度图像。
#因此输入通道数为1 同时class为10
class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ReLU = nn.ReLU()
        self.conv_1 = nn.Conv2d(in_channels=1,out_channels=96,kernel_size=11,stride=4)
        self.pool_2 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv_3 = nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,padding=2)
        self.pool_4 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv_5 = nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,padding=1)
        self.conv_6 = nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,padding=1)
        self.conv_7 = nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,padding=1)
        self.pool_8 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.flatten = nn.Flatten()
        self.fc_9 = nn.Linear(in_features=256*6*6,out_features=4096)
        self.fc_10 = nn.Linear(in_features=4096,out_features=4096)
        self.fc_11 = nn.Linear(in_features=4096,out_features=10)
    def forward(self,x):
        x = self.ReLU(self.conv_1(x))
        x = self.pool_2(x)
        x = self.ReLU(self.conv_3(x))
        x = self.pool_4(x)
        x = self.ReLU(self.conv_5(x))
        x = self.ReLU(self.conv_6(x))
        x = self.ReLU(self.conv_7(x))
        x = self.pool_8(x)
        x = self.flatten(x)
        x = self.ReLU(self.fc_9(x))
        x = F.dropout(x,p=0.5,training=self.training)
        x = self.ReLU(self.fc_10(x))
        x = F.dropout(x,p=0.5,training=self.training)
        x = self.fc_11(x)
        return x
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet().to(device)
    print(summary(model,input_size=(1,227,227)))

