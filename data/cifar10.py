import torch
from torchvision.datasets import CIFAR10
from PIL import Image
import numpy as np

from utils import make_imb_data, train_split


class CustomCIFAR10(CIFAR10):
    def __init__(self, class_num,max_num,imb_ratio,**kwds):
        super().__init__(**kwds)
        n_per_class = make_imb_data(max_num, class_num, imb_ratio, 'long')
        train_idx = train_split(self.targets, n_per_class)
        if train_idx is not None:
            self.data = self.data[train_idx, :, :, :]
            self.targets = np.array(self.targets)[train_idx]
        self.idxsPerClass = [np.where(np.array(self.targets) == idx)[0] for idx in range(10)]
        self.idxsNumPerClass = [len(idxs) for idxs in self.idxsPerClass]
        print(self.idxsNumPerClass)
    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(img).convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        target=self.targets[idx]
        return torch.stack(imgs),target,idx

