import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from resnet_variant import ResNetVariant
from utils import CIFAR10Dataset, unpickle, augmented_transform, TransformWrapper

# Define dataset directory and load CIFAR-10 batch files
data_dir = './data/cifar-10-batches-py'
batch_files = [os.path.join(data_dir, f'data_batch_{i}') for i in range(1, 6)]

data_list = []
labels_list = []
for file in batch_files:
    batch = unpickle(file)
    data_list.append(batch[b'data'])
    labels_list.extend(batch[b'labels'])
data_array = np.concatenate(data_list, axis=0)

# create the full dataset to split into training and validation sets
full_dataset = CIFAR10Dataset(data_array, labels_list, transform=None)
train_indices = list(range(45000))                   # 45000 training samples
val_indices = list(range(45000, 50000))              # 5000 validation samples
train_subset = Subset(full_dataset, train_indices)   
val_subset = Subset(full_dataset, val_indices)

# Apply transformations to training and validation datasets
train_dataset = TransformWrapper(train_subset, augmented_transform)
val_dataset = TransformWrapper(val_subset, augmented_transform)

# Create data loaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Define device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# print(torch.cuda.is_available())  # should return True
# print(torch.cuda.device_count())  # Check the number of available GPUs
# print(torch.cuda.get_device_name(0)) # Get the GPU name
# print(torch.__version__)        # Check PyTorch version
# print(torch.version.cuda)     # Check CUDA version

# Initialize model and move to device
model = ResNetVariant(num_classes=10).to(device)

# Define loss function, optimizer, and learning rate scheduler
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Training loop settings
start_epoch = 1
num_epochs = 300

# Training and validation loop
for epoch in range(start_epoch, num_epochs + 1):
    # ------------------------------ Training ------------------------------
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
    scheduler.step() # Update learning rate using scheduler(cosine annealing)

    # ------------------------------ Validation ------------------------------
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

    # Save model checkpoint every 10 epochs
    if epoch % 10 == 0 or epoch == num_epochs:
        torch.save(model.state_dict(), f'resnet_epoch{epoch}.pth')
        print(f"Epoch {epoch}: Model saved")
