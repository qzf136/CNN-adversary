import numpy as np
from torchvision import datasets
from adv_dataset import adv_dataset
from torch.utils.data import dataloader
import torch.optim as optim
from torchvision import transforms
import YeNet
import XuNet
import WuNet
import AlexNet
import YeNet2
import utils

batch_size = 128

transform = transforms.Compose([
    transforms.ToTensor(),
])

set1 = datasets.CIFAR10("./data", train=False, transform=transform, download=True)
loader1 = dataloader.DataLoader(set1, batch_size=batch_size, shuffle=True, num_workers=2)

train_dataset = adv_dataset(train=True)
test_dataset = adv_dataset(train=False)
train_loader = dataloader.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = dataloader.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


if __name__ == '__main__':
    # XuNet 检测对抗样本
    net = XuNet.XuNet().cuda()
    optimizer = optim.Adadelta(net.parameters(), lr=0.001, rho=0.95, eps=1e-8, weight_decay=5e-4)
    net = utils.train(train_loader, test_loader, epochs=100, path='./file/adv_XuNet.pt', net=net, optimizer=optimizer)

    # YeNet 检测对抗样本
    # net = YeNet.YeNet().cuda()
    # optimizer = optim.Adadelta(net.parameters(), lr=0.001, rho=0.95, eps=1e-8, weight_decay=5e-4)
    # net = utils.train(train_loader, test_loader, epochs=100, path='./file/adv_YeNet.pt', net=net, optimizer=optimizer)

    # WuNet 检测对抗样本
    # net = WuNet.init().cuda()
    # optimizer = optim.Adadelta(net.parameters(), lr=0.001, rho=0.95, eps=1e-8, weight_decay=5e-4)
    # net = utils.train(train_loader, test_loader, epochs=100, path='./file/adv_WuNet.pt', net=net, optimizer=optimizer)

    # YeNet 取消预处理层检测对抗样本
    # net = YeNet2.init().cuda()
    # optimizer = optim.Adadelta(net.parameters(), lr=0.001, rho=0.95, eps=1e-8, weight_decay=5e-4)
    # net = utils.train(train_loader, test_loader, epochs=100, path='./file/adv_YeNet2.pt', net=net, optimizer=optimizer)

    # AlexNet 检测对抗样本
    # net = AlexNet.init(3, 2).cuda()
    # optimizer = optim.Adadelta(net.parameters(), lr=0.001, rho=0.95, eps=1e-8, weight_decay=5e-4)
    # net = utils.train(train_loader, test_loader, epochs=100, path='./file/adv_AlexNet.pt', net=net, optimizer=optimizer)
