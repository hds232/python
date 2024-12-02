import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('../tensorboard/dataloader')

# 对数据集中每项数据的处理方式
tf = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# 没有文件时先将download设置为True，读取默认数据集
train_set = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, transform=tf)
test_set = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, transform=tf)

# 使用Dataloader进行取样打包
test_load = DataLoader(
    dataset=test_set, # 设置数据集
    batch_size=10, # 一次打包100个数据
    shuffle=True, # 取样时是否打乱顺序
    num_workers=0, # 是否打开多进程，报错时注意检查该选项
    drop_last=True, # 最后一组数据不足batch_size时是否舍弃
)

for idx, data in enumerate(test_load):
    imgs, targets = data
    writer.add_images('test2', img_tensor=imgs, global_step=idx)
writer.close()