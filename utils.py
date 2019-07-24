import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


# 训练模型
def train(train_loader, test_loader, epochs, path, net, optimizer):
    """
    训练网络模型
    :param train_loader:训练时使用的数据集
    :param test_loader: 测试时使用的数据集
    :param epochs: 训练的次数
    :param path:训练后得到的模型保存路径
    :param net:传进来训练的网络模型
    :return: 训练得到的模型
    """

    if net == None:
        print("模型为空")
        return

    loss_fuc = nn.CrossEntropyLoss().cuda()
    print("Training...")
    running_loss = 0.0
    for epoch in range(epochs):
        for step, (data, label) in enumerate(train_loader):
            data, label = Variable(data).cuda(), Variable(label).cuda()
            optimizer.zero_grad()
            out = net(data)
            loss = loss_fuc(out, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if step % 100 == 1:            # 每100步记录一遍loss值和正确率
                print("step =", step, "\tepoch =", epoch, "\tloss = {:.4f}".format(running_loss/100))
                running_loss = 0.0
        print("right ", cal_right_rate(test_loader, net))
        torch.save(net.state_dict(), path)  # 保存训练得到的网络模型
    print('Finish Training')
    return net


# 测试模型
def cal_right_rate(test_loader, net):
    """
    计算正确率的函数
    :param test_loader:传入的测试数据集,需要被DataLoader封装
    :param net:训练的网络
    :return:在测试集上的正确率
    """
    total = 0
    right = 0
    for step, (data, label) in enumerate(test_loader):
        data, label = Variable(data).cuda(), Variable(label).cuda()
        # print(np.shape(data))
        # print(np.shape(label))
        out = net(data)
        out_data = out.data
        _, predict = torch.max(out_data, 1)
        total += label.size()[0]
        right += (label == predict).sum().item()
    return right/total
