import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd

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
    

def generate_submission_csv(model, test_loader, filename="submission.csv", device="cuda"):
    """
    生成符合比赛格式的 CSV 文件
    参数:
        model: 训练好的 ResNetVariant 模型
        test_loader: 测试集 DataLoader
        filename: 生成的 CSV 文件名 (默认: "submission.csv")
    """
    model.eval()  # 设置模型为评估模式
    test_ids = []  # 存储图像 ID
    predicted_labels = []  # 存储预测的类别

    with torch.no_grad():  # 关闭梯度计算，加速推理
        for idx, (images, _) in enumerate(test_loader):  # CIFAR-10 测试集没有标签，所以 _ 忽略
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()  # 取最大概率的类别
            test_ids.extend(range(idx * test_loader.batch_size, idx * test_loader.batch_size + len(images)))
            predicted_labels.extend(preds)
            
    # 生成 DataFrame
    submission_df = pd.DataFrame({"ID": test_ids, "Labels": predicted_labels})
    submission_df.to_csv(filename, index=False)  # 保存为 CSV 文件
    print(f"✅ 预测结果已保存至 {filename}，请提交该文件。")