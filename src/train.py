import os
import pickle
import numpy as np
from PIL import Image

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm import tqdm
from torchvision import transforms
from model.resnet_variant import ResNetVariant


def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

augmented_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
])

class CIFAR10Dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.data[idx]
        r = row[0:1024].reshape(32, 32)
        g = row[1024:2048].reshape(32, 32)
        b = row[2048:3072].reshape(32, 32)
        img = np.stack([r, g, b], axis=2)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

data_dir = './data/cifar-10-batches-py'
batch_files = [os.path.join(data_dir, f'data_batch_{i}') for i in range(1, 6)]


data_list = []
labels_list = []
for file in batch_files:
    batch = unpickle(file)
    data_list.append(batch[b'data'])
    labels_list.extend(batch[b'labels'])
data_array = np.concatenate(data_list, axis=0)

full_dataset = CIFAR10Dataset(data_array, labels_list, transform=None)


train_indices = list(range(45000))
val_indices = list(range(45000, 50000))
train_subset = Subset(full_dataset, train_indices)
val_subset = Subset(full_dataset, val_indices)

class TransformWrapper(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

train_dataset = TransformWrapper(train_subset, augmented_transform)
val_dataset = TransformWrapper(val_subset, augmented_transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetVariant(num_classes=10).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

start_epoch = 1
num_epochs = 200

for epoch in range(start_epoch, num_epochs + 1):
    model.train()
    running_loss = 0.0
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} Training", leave=False)
    for images, labels in train_pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_pbar.set_postfix(loss=running_loss / (train_pbar.n + 1))
    scheduler.step()

    model.eval()
    correct, total = 0, 0
    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} Validation", leave=False)
    with torch.no_grad():
        for images, labels in val_pbar:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            val_pbar.set_postfix(acc=correct / total)
    acc = correct / total
    print(f'Epoch {epoch}, Loss: {running_loss / len(train_loader):.4f}, Acc: {acc:.4%}')

    if epoch % 10 == 0 or epoch == num_epochs:
        torch.save(model.state_dict(), f'resnet_epoch{epoch}.pth')
        print(f"Epoch {epoch}: 模型已保存。")
