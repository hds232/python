import cv2
from PIL import Image
from torchvision import transforms

img_path = '../dataset/test_data/train/ants/6240338_93729615ec.jpg'
# 返回PIL类型
img_PIL = Image.open(img_path)
# 返回numpy类型
img_cv = cv2.imread(img_path)

# 创建tensor类型转换实例，返回tensor对象
tf = transforms.ToTensor()
img_cv_tensor = tf(img_cv)
img_PIL_tensor = tf(img_PIL)

# 创建归一化实例，返回归一化tensor对象
norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1, 1, 1])
img_cv_tensor_norm = norm(img_cv_tensor)
img_PIL_tensor_norm = norm(img_PIL_tensor)

# 改变图片实例（PIL, tensor）的大小，返回一个相同类型的对象，最好是tensor
tre = transforms.Resize(size=(125, 125))
img_PIL_resize = tre(img_PIL)

# 对图片进行随机裁剪，输入图片实例随机，最好是tensor
trandom = transforms.RandomCrop((512, 126))

# compose实现上述功能的组合
# 将图片转换为tensor后改变图片大小，并将结果归一化
tcom = transforms.Compose([tf, tre, norm])
img_PIL_tcom = tcom(img_PIL)
img_cv_tcom = tcom(img_cv)
print(img_PIL_tcom)
print(img_cv_tcom)