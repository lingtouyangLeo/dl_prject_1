import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """基本残差块，每个块包含两个3x3卷积层。"""
    expansion = 1  # BasicBlock不改变通道数（扩张系数为1）

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        初始化基本残差块。
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数 (BasicBlock不会扩张通道，因此输出通道=设定通道)
            stride: 第一个卷积层的步幅。对于下采样块，stride=2；否则stride=1
            downsample: 下采样层，用于在跳跃连接中调整尺寸或通道数（如果需要）
        """
        super(BasicBlock, self).__init__()
        # 第一个3x3卷积，将输入通道转为out_channels，步幅stride控制下采样
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # 第二个3x3卷积，步幅固定为1（残差块中的下采样只在第一个卷积完成）
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 如果输入和输出的尺寸或通道不匹配，这里传入的downsample层用于调整identity
        self.downsample = downsample

    def forward(self, x):
        identity = x  # 保留输入tensor用于残差跳跃连接
        # 主分支的卷积操作
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # 如果需要下采样（调整通道或大小），通过downsample层转换identity
        if self.downsample is not None:
            identity = self.downsample(x)
        # 将主分支输出与identity相加，实现残差连接，然后再经过ReLU激活
        out += identity
        out = self.relu(out)
        return out


class ResNetVariant(nn.Module):
    """
    ResNet变体模型：
    - 总体结构基于ResNet-18，但通道数缩减为 (40, 80, 160, 320)，从而将参数量控制在约5百万以内 (约4.4M)。
    - 适用于CIFAR-10的32x32输入。
    """

    def __init__(self, num_classes=10):
        super(ResNetVariant, self).__init__()
        # 初始卷积层：输入3通道图像，输出40通道特征。使用3x3卷积，步幅1，保持32x32尺寸。
        self.in_channels = 40  # 当前特征图的通道数（将随模型进展而更新）
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        # 四个残差层组，每组输出通道数依次为 40, 80, 160, 320
        # 第一个残差层组（layer1）：输出通道40，不进行下采样（stride=1）
        self.layer1 = self._make_layer(out_channels=40, num_blocks=2, stride=1)
        # 第二个残差层组（layer2）：输出通道80，第一次卷积步幅2，实现下采样（将特征图尺寸从32x32降至16x16）
        self.layer2 = self._make_layer(out_channels=80, num_blocks=2, stride=2)
        # 第三个残差层组（layer3）：输出通道160，下采样（16x16降至8x8）
        self.layer3 = self._make_layer(out_channels=160, num_blocks=2, stride=2)
        # 第四个残差层组（layer4）：输出通道320，下采样（8x8降至4x4）
        self.layer4 = self._make_layer(out_channels=320, num_blocks=2, stride=2)
        # 全局平均池化层：将每个320通道的特征图（尺寸4x4）平均为1x1，从而大幅减少全连接层参数
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接分类层：将320通道的1x1特征向量映射到 num_classes（CIFAR-10中为10 类）
        self.fc = nn.Linear(320, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        """
        构建一个残差层组（包含若干BasicBlock）。
        参数:
            out_channels: 该层组中BasicBlock的输出通道数
            num_blocks: 该层组包含的BasicBlock数量
            stride: 该层组第一个BasicBlock卷积的步幅。stride=2表示下采样层，stride=1表示大小不变。
        """
        layers = []
        # 如需在第一个BasicBlock中进行下采样或通道数改变，则配置downsample层用于skip连接
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            # 使用1x1卷积调整通道数和步幅，以便跳跃连接匹配
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        # 添加第一个BasicBlock（可能包含下采样）
        layers.append(BasicBlock(self.in_channels, out_channels, stride=stride, downsample=downsample))
        # 更新当前通道数，以便后续Block使用
        self.in_channels = out_channels
        # 添加其余的BasicBlock，步幅为1（不再下采样）
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        # 将layers列表转换为Sequential模块
        return nn.Sequential(*layers)

    def forward(self, x):
        # 输入通过初始卷积、批归一化和ReLU激活
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 经过四个残差层组的前向传播
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # 全局平均池化，将特征图变为1x1
        out = self.avg_pool(out)
        # 展平Tensor，为全连接层做准备
        out = torch.flatten(out, 1)
        # 最后通过全连接层得到分类输出
        out = self.fc(out)
        return out


# 使用示例：创建模型并将其移动到GPU（如果可用），以支持在GPU上训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetVariant(num_classes=10).to(device)
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")  # 打印模型参数总量（验证≤5M）