import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),   # 随机裁剪
    transforms.RandomHorizontalFlip(),      # 随机水平翻转
    transforms.RandomRotation(15),          # 随机旋转
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),  # 颜色抖动
    transforms.ToTensor(),                  # 转换为张量
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])  # 归一化
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
])

class CIFAR10TestDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        if self.transform:
            img = self.transform(img)
        return img, idx
    
