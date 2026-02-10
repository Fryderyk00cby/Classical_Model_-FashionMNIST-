import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import GoogLeNet,Inception

def test_data_process():
    test_data = FashionMNIST(root = 'D:/code/MyModel/GoogLeNet/data',
                          train = False,
                          transform=transforms.Compose([transforms.Resize(size=224),transforms.ToTensor()]),
                          download=True)
    test_dataloader = Data.DataLoader(dataset=test_data,
                                       batch_size=1,
                                       shuffle =False,
                                       num_workers=0)
    return test_dataloader

def test_model_process(model,test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    test_correct = 0
    test_num = 0
    with torch.no_grad():
        for test_x,test_y in test_dataloader:
            test_x = test_x.to(device)
            test_y = test_y.to(device)
            model.eval()
            output = model(test_x)
            pred = torch.argmax(output,dim=1)
            test_correct += torch.sum(pred == test_y)
            test_num += test_y.size(0)
    test_acc = test_correct.double().item()/test_num
    print(f"Test Accuracy:{test_acc:.4f}")        

if __name__ == "__main__":
    test_dataloader = test_data_process()
    model = GoogLeNet(Inception)
    model.load_state_dict(torch.load("best_model.pth"))
    test_model_process(model,test_dataloader)
    #推理过程 搞100个试一试
    classes = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
               'Sandal','Shirt','Sneaker','Bag','Ankle boot']
    cnt = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    with torch.no_grad():
        for b_x,b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.eval()
            output = model(b_x)
            pred = torch.argmax(output,dim=1)
            result = pred.item()
            label = b_y.item()
            print(f"Predicted Label:{classes[result]},True Label:{classes[label]}")
            cnt += 1
            if cnt >=100:
                break
    print("Inference Done!")

    #正确率93.45%
