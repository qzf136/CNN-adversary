import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


# 卷积模块
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, with_bn=False):
        super(ConvBlock, self).__init__()
        # 卷积运算
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1)
        # relu激活函数
        self.relu = nn.ReLU()
        self.reset_parameters()

    def forward(self, x):
        # 前向运算
        return self.relu(self.conv(x))

    def reset_parameters(self):
        # 卷积模块权值初始化
        nn.init.xavier_uniform_(self.conv.weight)
        # 偏置初始化
        self.conv.bias.data.fill_(0.2)


# YeNet2总体
class YeNet2(nn.Module):
    # 构造网络
    def __init__(self):
        super(YeNet2, self).__init__()
        # 卷积模块
        self.block2 = ConvBlock(3, 30, 3)
        self.block3 = ConvBlock(30, 30, 3)
        self.block4 = ConvBlock(30, 30, 3)
        # pooling
        self.pool1 = nn.AvgPool2d(2, 2)
        self.block5 = ConvBlock(30, 32, 3,)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.block6 = ConvBlock(32, 32, 3,)
        self.pool3 = nn.AvgPool2d(2, 2)
        self.block7 = ConvBlock(32, 32, 3,)
        self.pool4 = nn.AvgPool2d(2, 2)
        self.block8 = ConvBlock(32, 16, 3)
        self.block9 = ConvBlock(16, 16, 3)
        # 线性激活层
        self.ip1 = nn.Linear(4 * 4 * 16, 2)
        # 根据条件重置参数
        self.reset_parameters()

    # 前向计算
    def forward(self, x):
        # 转换成float
        x = x.float()
        # 卷积运算
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        # pooling
        x = self.pool1(x)
        # print(np.shape(x))
        x = self.block5(x)
        x = self.pool2(x)
        # print(np.shape(x))
        x = self.block6(x)
        # x = self.pool3(x)
        x = self.block7(x)
        x = self.pool4(x)
        # print(np.shape(x))
        x = self.block8(x)
        x = self.block9(x)
        # print(np.shape(x))
        # 维度转换
        x = x.view(x.size(0), -1)
        # 全连接层
        x = self.ip1(x)
        return x

    def reset_parameters(self):
        for mod in self.modules():
            # 卷积层重置参数，这个根据卷积层，图像预处理层
            if isinstance(mod, ConvBlock):
                mod.reset_parameters()
            # 线性激活层
            elif isinstance(mod, nn.Linear):
                # 权值初始化
                nn.init.normal_(mod.weight, 0., 0.01)
                # 偏置初始化
                mod.bias.data.zero_()


def init():
    return YeNet2()