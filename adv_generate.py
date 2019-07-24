# 生成对抗样本

import torch
import numpy as np
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from advertorch.attacks import GradientSignAttack
from torch.autograd import Variable
from AlexNet import AlexNet

cifar10Path = './data'
batch_size = 10

AlexNet = AlexNet(3, 10).cuda()
AlexNet.load_state_dict(torch.load('./file/cifar_AlexNet.pt'))

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_data_set = torchvision.datasets.CIFAR10(root=cifar10Path, train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_data_set, batch_size=batch_size, shuffle=True)
test_data_set = torchvision.datasets.CIFAR10(root=cifar10Path, train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_data_set, batch_size=batch_size, shuffle=False)

# FGSM
adversary = GradientSignAttack(AlexNet, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.004, clip_min=0., clip_max=1., targeted=True)


def generate_sample(loader, path_data, path_label):
    data_adv = []
    label_adv = []
    i = 0
    for step, (true_data, true_label) in enumerate(loader):
        true_data = Variable(true_data).cuda()
        true_label = Variable(true_label).cuda()
        data = true_data
        i += 1
        if i % 2 == 1:
            for j in range(20):
                adv_targeted = adversary.perturb(data, (true_label + 1) % 10)
                data = adv_targeted
                # with torch.no_grad():
                #     outputs = cifarNet(adv_targeted)
                #     _, predicted = torch.max(outputs.data, 1)
            for k in range(batch_size):
                data_adv.append([data.cpu().detach().numpy()[k]])
                label_adv.append(1)
        else:
            for k in range(batch_size):
                data_adv.append([data.cpu().numpy()[k]])
                label_adv.append(0)
    print(np.shape(np.array(data_adv)))
    print(np.shape(np.array(label_adv)))
    np.save(path_data, np.array(data_adv))
    np.save(path_label, np.array(label_adv))


if __name__ == '__main__':
    generate_sample(train_loader, './file/train_data_adv.npy', './file/train_label_adv.npy')
    generate_sample(test_loader, './file/test_data_adv.npy', './file/test_label_adv.npy')