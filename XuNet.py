import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 定义整个网络
class XuNet(nn.Module):
    # 定义结构
    def __init__(self):
        super(XuNet, self).__init__()
        # 卷积层
        self.conv0 = nn.Conv2d(3, 1, 5, 1, bias=False)
        self.conv1 = nn.Conv2d(1, 8, 5, 1, bias=False)
        # bn层
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 5, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 5, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, 1, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        # 全连接层
        self.fc = nn.Linear(128, 2)

        # 图像预处理层的权值初始化
        lst = [[[-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1]]
               ] * 3
        kv = np.array(lst) * (1.0/12)
        kv = np.array([kv])
        kv = torch.Tensor(kv)
        self.conv0.weight = nn.Parameter(data=kv)

        # 其他层的权值和偏置初始化
        convcnt = 0
        for m in self.modules():
            # 卷积层
            if isinstance(m, nn.Conv2d):
                if (convcnt != 0):
                    # 正态分布
                    m.weight.data.normal_(0.0, 0.01)
                convcnt += 1
            # 全连接层
            elif isinstance(m,nn.Linear):
                # Xavier初始化
                nn.init.xavier_uniform_(m.weight, gain=1)

    # 前向传播
    def forward(self, x):
        out = self.conv0(x)
        # Group 1
        out = self.conv1(out)
        out = F.leaky_relu(out, -1.0)
        out = self.bn1(out)
        out = torch.tanh(out)
        # out = F.avg_pool2d(out, 5, 2)
        # Group 2
        out = self.conv2(out)
        out = self.bn2(out)
        out = torch.tanh(out)
        out = F.avg_pool2d(out, 5, 2)
        # Group 3
        out = self.conv3(out)
        out = self.bn3(out)
        out = F.relu(out)
        # out = F.avg_pool2d(out, 5, 2)
        # Group 4
        out = self.conv4(out)
        out = self.bn4(out)
        out = F.relu(out)
        # out = F.avg_pool2d(out, 5, 2)
        # Group 5
        out = self.conv5(out)
        out = self.bn5(out)
        out = F.relu(out)
        out = F.avg_pool2d(out, 32)
        # print(out)
        # FC
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = F.softmax(out, dim=1)
        return out


def init():
    return XuNet()
