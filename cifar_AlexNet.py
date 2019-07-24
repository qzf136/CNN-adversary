import numpy as np
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import dataloader
from torch import optim
from AlexNet import AlexNet
import utils

batch_size = 128
transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_data = datasets.CIFAR10("./data", train=True, transform=transform, download=True)
test_data = datasets.CIFAR10("./data", train=False, transform=transform, download=True)
train_loader = dataloader.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = dataloader.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


if __name__ == '__main__':
    net = AlexNet(3, 10).cuda()
    optimizer = optim.Adam(net.parameters())
    alex_net = utils.train(train_loader, test_loader, epochs=50, path='./file/cifar_AlexNet.pt', net=net, optimizer=optimizer)