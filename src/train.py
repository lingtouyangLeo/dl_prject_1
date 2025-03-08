import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import resnet
import os
from myutils import unpickle  
import numpy as np
import pandas as pd


# Select device (use GPU if available, otherwise use CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_batches = ["./data/data_batch_1", "./data/data_batch_2", 
                "./data/data_batch_3", "./data/data_batch_4", "./data/data_batch_5"]
test_batch = "./data/test_batch"

train_data, train_labels = [], []

# Load and concatenate all training batches
for batch in data_batches:
    batch_dict = unpickle(batch)
    train_data.append(batch_dict[b'data'])
    train_labels.extend(batch_dict[b'labels'])

# Convert to NumPy arrays and reshape to (N, 3, 32, 32)
train_data = np.vstack(train_data).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0  # Normalize to [0, 1]
train_labels = np.array(train_labels)

# Load test batch
test_dict = unpickle(test_batch)
test_data = test_dict[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0  # Normalize
test_labels = np.array(test_dict[b'labels'])


# Data augmentation strategies
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # Random cropping to 32x32
    transforms.RandomHorizontalFlip(),  # 50% chance of horizontal flip
    transforms.Normalize((0.5,), (0.5,))  # Normalize to range (-1, 1)
])

transform_test = transforms.Compose([
    transforms.Normalize((0.5,), (0.5,))
])

# Create PyTorch Datasets and DataLoaders
class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.tensor(self.data[idx], dtype=torch.float32)  # Convert to Tensor
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Create Dataset objects
train_dataset = CIFAR10Dataset(train_data, train_labels, transform=transform_train)
test_dataset = CIFAR10Dataset(test_data, test_labels, transform=transform_test)

# Create DataLoaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

model = resnet.ResNet18().to(device)  
print("Total Model Parameters:", sum(p.numel() for p in model.parameters()))

criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)  # Adam optimizer with L2 regularization

def train(model, trainloader, optimizer, criterion, device, epochs=10):
    model.train()  # Set to training mode
    for epoch in range(epochs):
        running_loss = 0.0
        correct, total = 0, 0
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Calculate accuracy for the current epoch
        train_acc = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss / len(trainloader):.4f} - Acc: {train_acc:.2f}%")

def test(model, testloader, criterion, device):
    model.eval()  # Set to evaluation mode
    test_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    # Calculate test accuracy
    test_acc = 100.0 * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")
    return test_acc


# for submission on kaggle
def generate_submission(model, testloader, device, output_file="submission.csv"):
    model.eval()  # Set model to evaluation mode
    predictions = []

    with torch.no_grad():
        for inputs, _ in testloader:  # Test data has no labels
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get predicted labels
            predictions.extend(predicted.cpu().numpy())

    # Create a DataFrame for submission
    submission_df = pd.DataFrame({"ID": range(len(predictions)), "Label": predictions})

    # Save as CSV
    submission_df.to_csv(output_file, index=False)
    print(f"Submission file saved: {output_file}")


if __name__ == '__main__':
    EPOCHS = 20  # Train for 20 epochs
    train(model, trainloader, optimizer, criterion, device, epochs=EPOCHS)

    test_accuracy = test(model, testloader, criterion, device)

    os.makedirs("./model", exist_ok=True)  # Ensure model directory exists
    model_path = f"./model/resnet18.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at: {model_path}")
    generate_submission(model, testloader, device)
