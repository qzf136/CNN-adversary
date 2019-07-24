import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ResNet-BasicBlock
class BasicBlock(nn.Module):
    expansion = 1

    # 结构定义
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # bn层
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # Shortcut
        self.shortcut = nn.Sequential()
        # 更改维数
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    # 前向传播
    def forward(self, x):
        # 第一层计算
        out = F.relu(self.bn1(self.conv1(x)))
        # 第二层计算
        out = self.bn2(self.conv2(out))
        # 快捷连接（Shortcut）
        out += self.shortcut(x)
        # relu激活
        out = F.relu(out)
        return out


# ResNet隐写分析
class WuNet(nn.Module):
    # 结构初始化
    def __init__(self, block, num_blocks, num_classes=2):
        super(WuNet, self).__init__()
        self.in_planes = 64
        # 预处理层
        self.conv0 = nn.Conv2d(3, 3, 5, 1, bias=False)
        # 普通卷积层
        self.conv1 = nn.Conv2d(3, 64, 7, 1, bias=False)
        # BN层
        self.bn1 = nn.BatchNorm2d(64)
        # ResNet部分
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        # 图像预处理层的权值初始化
        lst = [[[-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1]]
               ] * 3
        kv = np.array(lst) * (1.0 / 12)
        kv = np.array([kv, kv, kv])
        kv = torch.Tensor(kv)
        self.conv0.weight = nn.Parameter(data=kv)

        # ResNet模块的权值和偏置初始化
        convcnt = 0
        for m in self.modules():
            # print(m.__class__.__name__)
            if isinstance(m, nn.Conv2d):
                if convcnt != 0:
                    # 权值初始化，禁用偏置
                    m.weight.data.normal_(0.0, 0.01)
                    # m.bias.data.fill_(0)
                convcnt += 1

    # 根据参数构造ResNet层
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            # 追加ResNet子块
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    # 前向计算
    def forward(self, x):
        # 图像预处理层
        out = self.conv0(x)
        # 普通卷积层激活
        out = F.relu(self.bn1(self.conv1(out)))
        # pooling
        # out = F.max_pool2d(out, 3)
        # ResNet
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        # 全连接层
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        # Softmax层
        out = F.softmax(out, dim=1)
        return out


def init():
    # 整个网络初始化，返回模型
    return WuNet(BasicBlock, [2, 2, 1, 1])
