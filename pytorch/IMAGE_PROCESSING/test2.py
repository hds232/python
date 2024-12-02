import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# 使用前在conda该环境下添加tensorboard --logdir=logs --port=3602
# 控制台使用时挂起
writer = SummaryWriter('../tensorboard/logs')
# 获取图片组成的数组
img = np.array(Image.open('../dataset/test_data/train/ants/24335309_c5ea483bb8.jpg'))
# tensorboard添加图片
writer.add_image('test2', img, global_step=1, dataformats='HWC')
writer.close()