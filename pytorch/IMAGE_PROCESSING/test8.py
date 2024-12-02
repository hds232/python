import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

train_set = MNIST('../dataset/MNIST', train=True, transform=transforms.ToTensor())
test_set = MNIST('../dataset/MNIST', train=False)

device = torch.device('cuda')
class hand_classify(nn.Module):
    def __init__(self):
        super().__init__()
        self.get_features = nn.Sequential(
            nn.Conv2d(1, 10, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(10, 10, 4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(250, 50),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(50, 10),
            nn.LogSoftmax(dim=1)
        )
    def forward(self, input_data):
        tep = self.get_features(input_data)
        return self.classify(tep)

if __name__ == '__main__':
    moudle = hand_classify().to(device)
    # print(model.forward(torch.ones((1, 1, 28, 28)).cuda()))
    # 创建损失函数和优化器
    loss_f = nn.CrossEntropyLoss().to(device)
    learning_rate = 0.01
    optimizer = torch.optim.SGD(moudle.parameters(), lr=learning_rate)
    # 参数
    epoch = 100
    total_iter_num = 0
    # 训练
    moudle.train()
    for i in range(epoch):
        # 数据打包分类
        train_load = DataLoader(train_set, batch_size=100, shuffle=True, drop_last=True)
        for data in train_load:
            imgs, labels = data[0].to(device), data[1].to(device)
            output_data = moudle(imgs)
            # 损失函数
            loss = loss_f(output_data, labels)
            # 优化器
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_iter_num += 1
            if total_iter_num % 1000 == 0:
                print(f'iter_num:{total_iter_num}')
    torch.save(moudle.state_dict(), '../model/model.pth')