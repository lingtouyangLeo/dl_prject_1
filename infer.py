import torch
from torch.utils.data import DataLoader
from model.resnet_variant import ResNetVariant
from utils import val_transform
import pickle
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNetVariant(num_classes=10).to(device)
model.load_state_dict(torch.load('resnet_epoch10.pth'))
model.eval()

with open('data/cifar_test_nolabel.pkl', 'rb') as f:
    test_dict = pickle.load(f)

# 修改为字节字符串索引数据
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

with open('submission_epoch150.csv', 'w') as f:
    f.write('ID,Labels\n')
    for idx, label in predictions:
        f.write(f'{idx},{label}\n')

print('submission_epoch150.csv 文件已成功生成，可以提交到 Kaggle!')
