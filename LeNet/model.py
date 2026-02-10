import torch
from torch import nn
from torchsummary import summary


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,stride=1,padding=2) 
        self.sigmoid = nn.Sigmoid()
        self.pool_1 = nn.AvgPool2d(kernel_size=2,stride = 2)
        self.conv_2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1,padding=0)
        self.pool_2 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(400,120)
        self.fc_2 = nn.Linear(120,84)
        self.fc_3 = nn.Linear(84,10)
    def forward(self, x):
        x = self.sigmoid(self.conv_1(x))
        x = self.pool_1(x)
        x = self.sigmoid(self.conv_2(x))
        x = self.pool_2(x)
        x = self.flatten(x)
        x = self.sigmoid(self.fc_1(x))
        x = self.sigmoid(self.fc_2(x))
        x = self.fc_3(x)
        return x

        
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = LeNet().to(device)
#     print(summary(model,(1,28,28)))









