import torch.nn as nn


# AlexNet图像分类
class AlexNet(nn.Module):
    def __init__(self, n_channls, n_out):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
             # CIFAR10的图片为 [3,32,32],所以对原来的AlexNet网络进行了改造
            nn.Sequential(
                # 这是原来的卷积层
                # nn.Conv2d(n_channls, out_channels=96, kernel_size=11, stride=4),
                # 修改后的卷积层
                nn.Conv2d(n_channls, 96, 3),                  # 生成的数据 [96,30,30]
                nn.ReLU(),
                # 这是原来的池化层
                # nn.MaxPool2d(kernel_size=3,stride=2),
                # 修改后的池化层
                nn.MaxPool2d(2,2),                          # 生成的数据 [96,15,15]
                # 查阅资料发现LRB函数影响计算且效果并没有明显改善,所以去掉
                # nn.LocalResponseNorm(size=5)
            ),
            nn.Sequential(
                # 这是未修改的卷积层
                # nn.Conv2d(96,256,5,padding=2),
                # 这是修改后的卷积层
                nn.Conv2d(96, 256, 3, padding=1),              # 生成的数据 [256,15,15]
                nn.ReLU(),
                nn.MaxPool2d(3, 2),                         # 生成的数据 [256,6,6]
                # 去掉LRB函数
                # nn.LocalResponseNorm(5)
            ),
            nn.Sequential(
                nn.Conv2d(256, 384, 3, padding=1),          # 生成的数据 [384,6,6]
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(384, 384, 3, padding=1),          # 生成的数据 [384,6,6]
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(384, 256, 3, padding=1),          # 生成的数据 [256,6,6]
                nn.ReLU(),
                # 这是原来的池化层
                # nn.MaxPool2d(3, 2)
                # 这是修改后的池化层
                nn.MaxPool2d(2, 2)                          # 生成的数据 [256,3,3]

            )
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),        # 随机去掉一半的神经元
            # nn.Linear(6 * 6 * 256, 4096), # 原来的全连接层
            nn.Linear(3 * 3 * 256, 2048),   # 现在的全连接层
            nn.ReLU(),
            nn.Dropout(0.5),        # 随机去掉一半的神经元
            # nn.Linear(4096, 4096),        # 原来的全连接层
            nn.Linear(2048, 1024),          # 现在的全连接层
            nn.ReLU(),
            nn.Dropout(0.5),        # 随机去掉一半的神经元
            # nn.Linear(4096, n_out)        #原来的全连接层
            nn.Linear(1024, n_out)          # 现在的全连接层
        )


    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 3*3*256)
        out = self.fc(x)
        return out


def init(channel, dim_out):
    return AlexNet(channel, dim_out)