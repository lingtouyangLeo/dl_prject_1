import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """
    Basic building block for the ResNet variant.
    Each block consists of two convolutional layers with BatchNorm and ReLU activation,
    along with a residual connection that allows identity mapping.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):

        super(BasicBlock, self).__init__()
        # Convolutional layer 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Convolutional layer 2
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Downsample layer
        self.downsample = downsample

    def forward(self, x):
        identity = x                          # Store original input for residual connection
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:      # Apply downsample if necessary
            identity = self.downsample(x)
        out += identity                     # Add residual connection(skip connection)
        out = self.relu(out)
        return out

class ResNetVariant(nn.Module):
    """
    ResNet variant consists of an initial convolution layer followed by three residual layers,
    an adaptive average pooling layer, and a fully connected layer for classification.
    """
    def __init__(self, num_classes=10):
        super(ResNetVariant, self).__init__()
        self.in_channels = 64
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        # Residual layers
        self.layer1 = self._make_layer(out_channels=64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(out_channels=128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(out_channels=256, num_blocks=2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))     # Adaptive average pooling layer
        self.fc = nn.Linear(256, num_classes)            # Fully connected layer

    def _make_layer(self, out_channels, num_blocks, stride):
        """
        Creates a residual layer consisting of multiple BasicBlocks.
        If stride != 1 or input/output channels mismatch, a downsampling layer is added.
        """
        layers = []
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers.append(BasicBlock(self.in_channels, out_channels, stride=stride, downsample=downsample))
        self.in_channels = out_channels  # Update channel count for next layer
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# check the total number of parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetVariant(num_classes=10).to(device)
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
