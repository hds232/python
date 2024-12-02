import torch
from test6 import test_moudle, tf
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

test_set = CIFAR10('../dataset/CIFAR10', train=False, transform=tf)
test_loader = DataLoader(test_set, batch_size=100, shuffle=True, drop_last=True)

moudle = test_moudle().cuda()
moudle.load_state_dict(torch.load('../model/mymodel_advanced.pth', weights_only=True))
loss_f = torch.nn.CrossEntropyLoss().cuda()

# 测试开始
moudle.eval()
total_num = 0
with torch.no_grad() as t_no:
    for data in test_loader:
        imgs, targets = data[0].cuda(), data[1].cuda()
        output_data = moudle(imgs)
        loss = loss_f(output_data, targets)
        predict_label = output_data.argmax(axis=1)
        print(f'预测值:{predict_label}')
        print(f'实际值:{targets}')
        print(f'交叉熵:{loss}')
        total_num += torch.sum(predict_label == targets)
    print(f'成功率:{total_num/len(test_set)}')
