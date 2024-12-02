import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

# 转换方式组合
tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
     ])
# 定义优化模式
device = torch.device('cuda')
class test_moudle(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 使用Sequential函数创建一个模型
        self.moudle = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 2, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 32, 3, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 4, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Dropout(),
            torch.nn.Linear(1600, 64),
            torch.nn.Dropout(),
            torch.nn.Linear(64, 10),
            torch.nn.LogSoftmax(dim=1)
        )
    def forward(self, input_data:torch.tensor) -> torch.tensor:
        output_data = self.moudle(input_data)
        return output_data

if __name__ == '__main__':
    # 创建模型实例
    moudle = test_moudle().to(device)
    # 读取数据文件
    train_set = CIFAR10(
        '../dataset/CIFAR10', train=True,
        transform=tf,
    )
    # # 创建损失函数
    # loss_f = torch.nn.CrossEntropyLoss().to(device)
    # # 创建优化器
    # learning_rate = 1e-2
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # # 定义训练的总轮次和迭代总次数
    # epoch = 100
    # total_iter_num = 0
    # # 训练过程开始
    # model.train()
    # for count in range(epoch):
    #     # 每轮随机抓取图片
    #     train_load = DataLoader(train_set, batch_size=100, shuffle=True, drop_last=True)
    #     for data in train_load:
    #         imgs, lables = data[0].to(device), data[1].to(device)
    #         output_data = model(imgs)
    #         # 计算损失函数
    #         loss = loss_f(output_data, lables)
    #         # 优化器根据梯度优化模型
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         # 统计训练总次数
    #         total_iter_num += 1
    #         if total_iter_num % 1000 == 0:
    #             print(f'迭代{total_iter_num}次')
    # # 保存模型
    # torch.save(model.state_dict(), 'model/mymodel_advanced.pth')
    from torch.utils.tensorboard import SummaryWriter
    print(train_set[0][0].shape)
    test_data = torch.ones(size=(1, 3, 32, 32))
    writer = SummaryWriter('../tensorboard/logs')
    writer.add_graph(moudle, test_data.to(device))
    writer.close()
