import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # To keep the dimensions consistent, we need to add a shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  
        return F.relu(out)


class ResNet(nn.Module):
    # num_blocks: number of blocks in each layer;
    # num_classes: number of classes in the dataset(10 in CIFAR-10).
    def __init__(self, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 32  # input is 32x32X3 imgs(rgb)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(32)
        
        self.layer1 = self._make_layer(32, num_blocks[0], stride=1) # 32 channels
        self.layer2 = self._make_layer(64, num_blocks[1], stride=2) # 64 channels
        self.layer3 = self._make_layer(128, num_blocks[2], stride=2) # 128 channels
        
        self.avgpool = nn.AdaptiveAvgPool2d(1) # Global Average Pooling
        self.fc = nn.Linear(128, num_classes) # output layer
 
    # create residual blocks in each layer
    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels, stride if i == 0 else 1))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)

def ResNet8():
    return ResNet([1, 1, 1]) # 8 layers

def ResNet10():
    return ResNet([1, 2, 2]) # 10 layers

def ResNet14():
    return ResNet([2, 2, 2]) # 14 layers

def ResNet18():
    return ResNet([2, 2, 2, 2]) # 18 layers