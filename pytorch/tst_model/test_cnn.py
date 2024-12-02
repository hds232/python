import os    #os库用于文件管理，打开、删除文件等
import torch
from PIL import Image #图像处理库，支持多种格式的图像
import torch.nn as nn #导入torch中的nn模块
from torch import optim #调用torch中的优化器
from torch.utils.data import Dataset #dataset数据集类，用于表示一个数据集
from torch.utils.data import DataLoader #用于批量加载数据的工具，可以指定加载的参数
# from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms as tfs
import torch.nn.functional as F #无状态的函数，需要显式传递参数。


# 定义数据集
class Data_set(Dataset):
    # 初始化图片，可更改
    DATA_CHANGE = tfs.Compose((
        tfs.ToTensor(),
        tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化,
    ))

    # 定义数据集相关方法
    def __init__(self, path):
        # 图片地址和名称
        self.path = path
        self.img_name = os.listdir(self.path)

    def __getitem__(self, index) -> tuple:
        img = Image.open(os.path.join(self.path, self.img_name[index]))
        img = img.resize((32, 32), Image.LANCZOS)
        label = self.img_name[index].split('_')[0]
        return self.DATA_CHANGE(img), int(label) - 1

    def __len__(self):
        return len(self.img_name)
#加载数据集
trainset = Data_set('./lsa16_raw')

# 类别数
num_classes = 16

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  #卷积层 1 输入通道数: 3（对应 RGB 图像），输出通道数: 32（输出特征图的数量），卷积核大小: 3x3
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) #池化层，池化核大小: 2x2，每次移动2个像素
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) #卷积层 2 输入通道数: 32（来自卷积层 1），输出通道数: 64，卷积核大小: 3x3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) #卷积层 3 输入通道数: 64（来自卷积层 2），输出通道数: 128，卷积核大小: 3x3
        self.fc1 = nn.Linear(128 * 4 * 4, 256)  # 图像大小32x32，全连接层输入特征数: 128 * 4 * 4（从最后一个池化层输出的特征图展平后的维度），输出特征数: 256，将卷积层提取的特征映射到更高维空间，为后续分类做准备
        self.fc2 = nn.Linear(256, num_classes) #输入特征数: 256（来自 fc1 的输出），输出特征数: num_classes（分类任务中的类别数）

    def forward(self, x):  #定义了数据如何通过网络进行前向传播。
        x = self.pool(F.relu(self.conv1(x))) #通过第一卷积层和激活函数 ReLU
        x = self.pool(F.relu(self.conv2(x))) #依次通过第二卷积层、ReLU 和池化。
        x = self.pool(F.relu(self.conv3(x))) #依次通过第三卷积层、ReLU 和池化。
        x = x.view(-1, 128 * 4 * 4)  # 将特征图展平为一维
        x = F.relu(self.fc1(x))  #通过第一个全连接层和 ReLU。
        x = self.fc2(x)   #通过第二个全连接层，输出分类结果。
        return x

# 创建模型实例
model = CNNModel()
# 选择GPU还是CPU训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    trainloader = DataLoader(
        dataset=trainset,
        batch_size=64,  # 每个批次加载数据集中图片的个数
        shuffle=True,  # 设置是否随机
        drop_last=True
    )
    for imgs, labels in trainloader:
        imgs, labels = imgs.to(device), labels.to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}')


# 评估模型
model.eval()
correct = 0
total = 0
testloader = DataLoader(
    dataset=trainset,
    batch_size=1,  # 每个批次加载数据集中图片的个数
    shuffle=True,  # 设置是否随机
    drop_last=True
)
with torch.no_grad():
    for imgs, labels in testloader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'成功率: {100 * correct / total:.2f}%')
torch.save(model.state_dict(), 'model_new.pth')