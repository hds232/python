import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch.utils.tensorboard import SummaryWriter

# 读取数据
tf = transforms.Compose([
     transforms.ToTensor()
])
test_set = CIFAR10(root='./CIFAR10', train=False, transform=tf)
test_load = DataLoader(dataset=test_set, batch_size=100, num_workers=2)

class test_moudle(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个卷积层
        self.conv1 = nn.Conv2d(
            in_channels=3, # 卷积核输入通道
            out_channels=3, # 卷积核输出通道
            kernel_size=4, # 卷积核大小
            stride=1, # 卷积步长
            padding=0, # 填充宽度
            padding_mode='zeros', # 填充方法
        )
        # 定义一个池化层，输入tensor数据类型必须是torch.float类型
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=3, # 定义池化核的大小
            ceil_mode=True, # 池化大小不足时依旧取最大值
            stride=None, # 池化步长
            padding=0 # 填充宽度
        )
        # 定义一个非线性激活层，不就地执行
        self.relu1 = nn.ReLU(inplace=False)
        self.sigmoid1 = nn.Sigmoid()
        # 定义展开层
        self.flatten1 = nn.Flatten(start_dim=0, end_dim=3)
        # 定义一个线型层
        self.linear1 = nn.Linear(
            in_features=19200, # 输入通道大小
            out_features=10, # 输出通道大小
        )

    def forward(self, input_data):
        output_conv1 = self.conv1(input_data)
        output_maxpool1 = self.maxpool1(output_conv1)
        output_relu1 = self.relu1(output_maxpool1)
        output_sigmoid1 = self.sigmoid1(output_relu1)
        output_flatten1 = self.flatten1(output_sigmoid1)
        output_linear1 = self.linear1(output_flatten1)
        return output_linear1

if __name__ == '__main__':
    writer = SummaryWriter('../tensorboard/test_moudle')
    test_obj = test_moudle()
    input_data = torch.ones((64, 3, 32, 32))
    writer.add_graph(test_obj, input_data)
    # 测试数据输出
    # for idx, data in enumerate(test_load):
    #     imgs, _ = data
    #     output = test_obj(imgs)
    #     writer.add_images(
    #         tag='test',
    #         img_tensor=output,
    #         global_step=idx
    #     )
    writer.close()