import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from resnet_variant import ResNetVariant
import pickle
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型加载（使用最佳权重）
model = ResNetVariant(num_classes=10).to(device)
model.load_state_dict(torch.load('resnet_epoch100.pth'))  # 确保这是你最好的权重
model.eval()

# 测试集的数据预处理（务必与训练时完全一致）
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.247, 0.243, 0.261))
])

with open('data/cifar_test_nolabel.pkl', 'rb') as f:
    test_dict = pickle.load(f)

test_images = test_dict[b'data']
test_ids = test_dict[b'ids']

class CIFAR10TestDatasetWithID(torch.utils.data.Dataset):
    def __init__(self, images, ids, transform=None):
        self.images = images
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.ids[idx]

test_set = CIFAR10TestDatasetWithID(test_images, test_ids, transform=val_transform)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

predictions = []
with torch.no_grad():
    for images, indices in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        for idx, pred in zip(indices.tolist(), preds.tolist()):
            predictions.append((idx, pred))

predictions.sort(key=lambda x: x[0])

with open('submission_epoch100.csv', 'w') as f:
    f.write('ID,Labels\n')
    for idx, label in predictions:
        f.write(f'{idx},{label}\n')

print('submission_epoch100.csv 已成功生成！请立即提交到 Kaggle。')