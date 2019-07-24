import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

# 图像预处理层，先用numpy载入初始化的30个SRM核的数据
SRM_npy = np.load('./file/kernel.npy')


# 图像预处理层的类
class SRM_conv2d(nn.Module):
    def __init__(self, stride=1, padding=2):
        super(SRM_conv2d, self).__init__()
        # 输入1层（图像）
        self.in_channels = 3
        # 输出30层，因为是有30个卷积核，分别进行计算得到30个
        self.out_channels = 30
        # 设置卷积核大小
        self.kernel_size = (5, 5)
        # 设置步长
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
        # 设置padding
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        # 卷积膨胀
        self.dilation = (1,1)
        # 转置
        self.transpose = False
        # padding
        self.output_padding = (0,)
        # 分组，默认设置成1组
        self.groups = 1
        # 设置预处理层卷积核权值为30个5*5的Tensor，此时只是设置，并没有初始化
        self.weight = Parameter(torch.Tensor(30, 3, 5, 5),requires_grad=True)
        # 设置预处理层卷积核偏置为30个Tensor，此时只是设置，并没有初始化
        self.bias = Parameter(torch.Tensor(30),requires_grad=True)
        # 重置上面值的大小
        self.reset_parameters()

    def reset_parameters(self):
        # 将上面加载的SRM核，载入到self.weight中
        self.weight.data.numpy()[:] = SRM_npy
        # 默认设置偏置为0
        self.bias.data.zero_()

    def forward(self, input):
        # 前向计算
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


# 卷积模块
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, with_bn=False):
        super(ConvBlock, self).__init__()
        # 卷积运算
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1)
        # relu激活函数
        self.relu = nn.ReLU()
        # 传递启用BN层的参数
        self.with_bn = with_bn
        # 如果启用BN层参数开启
        if with_bn:
            # BN计算
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            # 若未开启，则不进行BN层计算，直接传递运算结果
            self.norm = lambda x: x
        self.reset_parameters()

    def forward(self, x):
        # 前向运算
        return self.norm(self.relu(self.conv(x)))

    def reset_parameters(self):
        # 卷积模块权值初始化
        nn.init.xavier_uniform_(self.conv.weight)
        # 偏置初始化
        self.conv.bias.data.fill_(0.2)


# YeNet总体
class YeNet(nn.Module):
    # 构造网络
    def __init__(self):
        super(YeNet, self).__init__()
        # 图像预处理层
        self.preprocessing = SRM_conv2d()
        # 卷积模块
        self.block2 = ConvBlock(30, 30, 3)
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
        # 预处理
        x = self.preprocessing(x)
        x = F.relu(x)
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
            if isinstance(mod, SRM_conv2d) or isinstance(mod, ConvBlock):
                mod.reset_parameters()
            # 线性激活层
            elif isinstance(mod, nn.Linear):
                # 权值初始化
                nn.init.normal_(mod.weight, 0., 0.01)
                # 偏置初始化
                mod.bias.data.zero_()

def init():
    return YeNet()