import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from resnet_variant import ResNetVariant
import pickle
from utils import CIFAR10TestDatasetWithID

# Define device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained ResNet model and set to evaluation mode
model = ResNetVariant(num_classes=10).to(device)
model.load_state_dict(torch.load('resnet_epoch200.pth'))
model.eval()

# Define normalization transformation for test images
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.247, 0.243, 0.261))
])

# Load test dataset (without labels) from pickle file
with open('data/cifar_test_nolabel.pkl', 'rb') as f:
    test_dict = pickle.load(f)

test_images = test_dict[b'data']  # Test images
test_ids = test_dict[b'ids']      # Corresponding image IDs

# Create test dataset and data loader
test_set = CIFAR10TestDatasetWithID(test_images, test_ids, transform=val_transform)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

# Perform inference on test dataset
predictions = []
with torch.no_grad():
    for images, indices in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        for idx, pred in zip(indices.tolist(), preds.tolist()):
            predictions.append((idx, pred))

# Sort predictions by image ID to maintain correct order
predictions.sort(key=lambda x: x[0])

# Save predictions to CSV file
with open('submission_corrected.csv', 'w') as f:
    f.write('ID,Labels\n')
    for idx, label in predictions:
        f.write(f'{idx},{label}\n')

print('Output successfully saved to submission.csv')
