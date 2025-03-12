from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pickle
import numpy as np

# Function to load CIFAR-10 data from pickle files
def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

# Data augmentation transformations for training
augmented_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),               # Randomly crop the image
    transforms.RandomHorizontalFlip(),                  # Randomly flip the image horizontally
    transforms.RandomRotation(15),                      # Randomly rotate the image by 15 degrees
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),         # Randomly change the brightness, contrast, and saturation
    transforms.ToTensor(),                              # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])  # Normalize the image    
])

# class for CIFAR-10 dataset
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
    
# Wrapper class to apply transformations to the dataset# Wrapper class to apply transformations to a dataset subset
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

# Custom dataset class for handling test images with IDs
class CIFAR10TestDatasetWithID(Dataset):
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