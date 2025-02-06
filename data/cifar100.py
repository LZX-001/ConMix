import torch
from torchvision.datasets import CIFAR100
from PIL import Image
import numpy as np

from utils import make_imb_data, train_split


class CustomCIFAR100(CIFAR100):
    def __init__(self, class_num,max_num,imb_ratio,**kwds):
        super().__init__(**kwds)
        self.targets = superclass(self.targets)
        n_per_class = make_imb_data(max_num,class_num,imb_ratio,'long')
        train_idx = train_split(self.targets,n_per_class)
        if train_idx is not None:
            self.data = self.data[train_idx,:,:,:]
            self.targets = np.array(self.targets)[train_idx]
        class_num = len(np.unique(self.targets))
        self.idxsPerClass = [np.where(np.array(self.targets) == idx)[0] for idx in range(class_num)]
        self.idxsNumPerClass = [len(idxs) for idxs in self.idxsPerClass]
        print(self.idxsNumPerClass)
    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(img).convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        target = self.targets[idx]
        return torch.stack(imgs),target,idx
def superclass(targets):
    targets = np.array(targets)
    super_classes = [
        [72, 4, 95, 30, 55],
        [73, 32, 67, 91, 1],
        [92, 70, 82, 54, 62],
        [16, 61, 9, 10, 28],
        [51, 0, 53, 57, 83],
        [40, 39, 22, 87, 86],
        [20, 25, 94, 84, 5],
        [14, 24, 6, 7, 18],
        [43, 97, 42, 3, 88],
        [37, 17, 76, 12, 68],
        [49, 33, 71, 23, 60],
        [15, 21, 19, 31, 38],
        [75, 63, 66, 64, 34],
        [77, 26, 45, 99, 79],
        [11, 2, 35, 46, 98],
        [29, 93, 27, 78, 44],
        [65, 50, 74, 36, 80],
        [56, 52, 47, 59, 96],
        [8, 58, 90, 13, 48],
        [81, 69, 41, 89, 85],
    ]
    import copy
    copy_targets = copy.deepcopy(targets)
    for i in range(len(super_classes)):
        for j in super_classes[i]:
            targets[copy_targets == j] = i
    return targets

