import os
from PIL import Image
from torch.utils.data import Dataset
# from torch.cuda import is_available

class Data_set(Dataset):
    def __init__(self, root_dir:str, label_dir:str) -> None:
        # 所有数据根目录
        self.root_dir = root_dir
        # 标签信息目录
        self.label_dir = label_dir
        # 数据目录
        self.img_name = os.listdir(os.path.join(self.root_dir, self.label_dir))

    def __getitem__(self, item) -> tuple:
        # 读取图像
        img = Image.open(os.path.join(self.root_dir, self.label_dir, self.img_name[item]))
        # 返回标签
        return img, self.label_dir

    def __len__(self):
        return len(self.img_name)

if __name__ == '__main__':
    ants_set = Data_set(root_dir='../dataset/test_data/train', label_dir='ants')
    bees_set = Data_set(root_dir='../dataset/test_data/train', label_dir='bees')
    print(len(ants_set))