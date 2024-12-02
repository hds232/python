import os
import torch
import pprint as p
from PIL import Image
import torch.nn as nn
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms as tfs

def clear_files(path:str):
    if os.path.exists(path):
        for filename in os.listdir(path):
            filepath = os.path.join(path, filename)
            if os.path.isfile(filepath):
                os.remove(filepath)

# 定义数据集
class Data_set(Dataset):
    # 初始化图片，可更改
    DATA_CHANGE = tfs.Compose((
        tfs.ToTensor(),
        # tfs.Normalize([0.5], [0.5]),
    ))
    # 定义数据集相关方法
    def __init__(self, path):
        # 图片地址和名称
        self.path = path
        self.img_name = os.listdir(self.path)
    def __getitem__(self, index) -> tuple:
        img = Image.open(os.path.join(self.path, self.img_name[index]))
        img = img.resize((64, 64), Image.LANCZOS)
        label = self.img_name[index].split('_')[0]
        return self.DATA_CHANGE(img), int(label) - 1
    def __len__(self):
        return len(self.img_name)

# 定义模型
class Model(nn.Module):
    # 相关参数
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 数据集
    trnset = Data_set('./lsa16_raw')
    tstset = trnset
    # 损失函数
    LOSS_F = nn.CrossEntropyLoss().to(DEVICE)
    # 相关方法
    def choose(self):
        return self.to(Model.DEVICE)
    def __init__(self):
        super(Model, self).__init__()
        self.batch = nn.BatchNorm2d(3)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4,  padding=6, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(stride=8, kernel_size=8)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=2, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4)
        )
        self.ln = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(64, 32),
            nn.Dropout(),
            nn.Linear(32 ,16),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.ln(x)
        return x
    def start_train(self, learning_rate = 1e-3, epoch = 100, dataloader_batch_size = 1, tensorboard = False):
        if tensorboard:
            writer = SummaryWriter('log')
        # 定义优化器，可在此更改
        OPTIM = optim.SGD(self.parameters(), learning_rate)
        # OPTIM = optim.Adam(self.parameters(), learning_rate)
        # 定义总的迭代次数
        total_running_num = 0
        # 开始训练
        self.train()
        for _ in range(epoch):
            # 每次循环开始时随机抓取图片
            trn_loader = DataLoader(
                dataset=Model.trnset,
                batch_size=dataloader_batch_size,
                shuffle=True, # 设置是否随机
                drop_last=True
            )
            for data in trn_loader:
                imgs, labels = data
                imgs = imgs.to(Model.DEVICE)
                labels = labels.to(Model.DEVICE)
                # 计算输出
                output_info = self(imgs)
                # 计算损失函数
                loss = self.LOSS_F(output_info, labels)
                # 反向传播
                OPTIM.zero_grad()
                loss.backward()
                # 梯度裁剪,使用Adam时可注释，防止干扰Adam本身预防
                # 梯度爆炸的机制
                clip_grad_norm_(self.parameters(), 10)
                # 优化参数
                OPTIM.step()
                total_running_num += 1
                if total_running_num % 1000 == 0:
                    print(f'迭代次数:{total_running_num}')
                if tensorboard:
                    writer.add_scalar('loss', loss, total_running_num)
        if tensorboard:
            writer.close()
    def start_test(self, dataloader_batch_size=10):
        # 测试数据集
        tst_loader = DataLoader(
            dataset=Model.tstset,
            batch_size=dataloader_batch_size,
            shuffle=True,
            drop_last=True
        )
        # 测试总次数
        total_num = 0
        # 测试开始
        self.eval()
        with torch.no_grad() as t_nog:
            for data in tst_loader:
                imgs, labels = data
                imgs = imgs.to(Model.DEVICE)
                labels = labels.to(Model.DEVICE)
                output_info = self(imgs)
                # 计算损失函数
                loss = self.LOSS_F(output_info, labels)
                predict_labels = output_info.argmax(axis=1)
                # 输出
                print(f'预测值:{predict_labels}')
                print(f'实际值:{labels}')
                print(f'交叉熵:{loss}')
                total_num += torch.sum(predict_labels == labels)
            print(f'成功率:{total_num / len(Model.tstset)}')
    # 输出参数
    def print_params(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                p.pprint(name)
                p.pprint(param.data)
    # 加载模型            
    def load_model(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))
    # 保存模型
    def save_model(self, path):
        torch.save(self.state_dict(), path)

if __name__ == '__main__':
    clear_files('./path')
    obj = Model().choose()
    # 可在此设置学习率，具体参数见定义
    obj.start_train(
        epoch=10000, # 迭代次数
        learning_rate=0.001, #学习率
        dataloader_batch_size=10, #每次循环抓取的图片数量
        tensorboard=True
    )
    obj.start_test()
    obj.print_params()
    obj.save_model('./model.pth')