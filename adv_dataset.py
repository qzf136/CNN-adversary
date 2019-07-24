import torch
import numpy as np
from torch.utils.data import Dataset

# 封装对抗样本
class adv_dataset(Dataset):

    def __init__(self, train, transform=None):
        self.transform = transform
        self.train = train

        if self.train:
            self.train_data = np.load('./file/train_data_adv.npy')      # 训练样本图片
            self.train_label = np.load('./file/train_label_adv.npy')    # 训练样本标签
        else:
            self.test_data = np.load('./file/test_data_adv.npy')    # 测试样本图片
            self.test_label = np.load('./file/test_label_adv.npy')  # 测试样本标签


    def __getitem__(self, index):
        if self.train:
            img, target = torch.tensor(self.train_data[index][0]), torch.tensor(self.train_label[index]).long()
        else:
            img, target = torch.tensor(self.test_data[index][0]), torch.tensor(self.test_label[index]).long()
        if self.transform is not None:
            img = self.transform(img)
        return img, target


    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
