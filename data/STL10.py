# from __future__ import print_function

import torch
from PIL import Image
import os
import os.path
import numpy as np
from torch.utils import data
from torchvision.datasets import STL10
import matplotlib.pyplot as plt

from utils import make_imb_data, train_split


class CustomSTL10(STL10):
    def __init__(self,class_num,max_num,imb_ratio,**kwds):
        super().__init__(**kwds)
        data, labels = self.concatenateData()
        self.data = data
        self.labels = labels
        n_per_class = make_imb_data(max_num, class_num, imb_ratio, 'long')
        train_idx = train_split(self.labels, n_per_class)
        if train_idx is not None:
            self.data = self.data[train_idx, :, :, :]
            self.labels = np.array(self.labels)[train_idx]
        self.idxsPerClass = [np.where(np.array(self.labels) == idx)[0] for idx in range(10)]
        self.idxsNumPerClass = [len(idxs) for idxs in self.idxsPerClass]
        print(self.idxsNumPerClass)
    def concatenateData(self):
        train_dataset = STL10(root='/nas/datasets', split='train')
        test_dataset = STL10(root='/nas/datasets',split='test')
        train_dataset.data = np.concatenate([train_dataset.data,test_dataset.data],axis=0)
        train_dataset.labels = np.concatenate([train_dataset.labels,test_dataset.labels],axis=0)
        return train_dataset.data, train_dataset.labels
    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(np.transpose(img, (1, 2, 0))).convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        label = self.labels[idx]
        return torch.stack(imgs), label, idx