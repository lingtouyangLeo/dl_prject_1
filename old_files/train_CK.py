import torch
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from utils import train_transform, val_transform
from resnet_variant import ResNetVariant

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# print(torch.cuda.is_available())  # 应该返回 True
# print(torch.cuda.device_count())  # 检查可用的 GPU 数量
# print(torch.cuda.get_device_name(0))  # 获取 GPU 名称
# print(torch.__version__)
# print(torch.version.cuda)  # 检查 CUDA 版本
# 加载数据集
full_train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
train_set, val_set = random_split(full_train_set, [45000, 5000])
train_set.dataset.transform = train_transform
val_set.dataset.transform = val_transform

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
val_loader = DataLoader(val_set, batch_size=128, shuffle=False)

# 定义模型
model = ResNetVariant(num_classes=10).to(device)

# 加载第90个epoch的权重
# model.load_state_dict(torch.load('resnet_epoch150.pth'))

criterion = torch.nn.CrossEntropyLoss()

# 微调通常使用较低的学习率，比如0.01或0.001
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# 从第91个epoch开始训练，训练到第190个epoch
start_epoch = 0
num_epochs = 100

for epoch in range(start_epoch, num_epochs):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = torch.nn.CrossEntropyLoss()(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    scheduler.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Acc: {acc:.4%}')

    # 定期保存训练的模型权重，比如每10个epoch保存一次
    if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
        torch.save(model.state_dict(), f'resnet_epoch{epoch+1}.pth')
        print(f"Epoch {epoch+1}: 模型已保存。")