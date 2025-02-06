import torch
import torch.nn as nn
import os
import time
import numpy as np
import random
import torch.nn.functional as F
import re
from pdb import set_trace


def train_split(labels, n_per_class):
    labels = np.array(labels)
    train_idxs = []
    length = len(n_per_class)
    for i in range(length):
        idxs = np.where(labels == i)[0]
        train_idxs.extend(idxs[:n_per_class[i]])
    return train_idxs


def make_imb_data(max_num, class_num, gamma, imb):
    if imb == 'long':
        mu = np.power(1 / gamma, 1 / (class_num - 1))
        class_num_list = []
        for i in range(class_num):
            if i == (class_num - 1):
                class_num_list.append(int(max_num / gamma))
            else:
                class_num_list.append(int(max_num * np.power(mu, i)))
        print(class_num_list)
    if imb == 'step':
        class_num_list = []
        for i in range(class_num):
            if i < int((class_num) / 2):
                class_num_list.append(int(max_num))
            else:
                class_num_list.append(int(max_num / gamma))
        print(class_num_list)
    return list(class_num_list)
def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class logger(object):
    def __init__(self, path, log_name="log.txt", local_rank=0):
        self.path = path
        self.local_rank = local_rank
        self.log_name = log_name

    def info(self, msg):
        if self.local_rank == 0:
            print(msg)
            with open(os.path.join(self.path, self.log_name), 'a') as f:
                f.write(msg + "\n")


def fix_bn(model, fixto):
    if fixto == 'nothing':
        # fix none
        # fix previous three layers
        pass
    elif fixto == 'layer1':
        # fix the first layer
        for name, m in model.named_modules():
            if not ("layer2" in name or "layer3" in name or "layer4" in name or "fc" in name):
                m.eval()
    elif fixto == 'layer2':
        # fix the previous two layers
        for name, m in model.named_modules():
            if not ("layer3" in name or "layer4" in name or "fc" in name):
                m.eval()
    elif fixto == 'layer3':
        # fix every layer except fc
        # fix previous four layers
        for name, m in model.named_modules():
            if not ("layer4" in name or "fc" in name):
                m.eval()
    elif fixto == 'layer4':
        # fix every layer except fc
        # fix previous four layers
        for name, m in model.named_modules():
            if not ("fc" in name):
                m.eval()
    else:
        assert False


def gatherFeatures(features, local_rank, world_size):
    features_list = [torch.zeros_like(features) for _ in range(world_size)]
    torch.distributed.all_gather(features_list, features)
    features_list[local_rank] = features
    features = torch.cat(features_list)
    return features


def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask
def nt_xent(x, t=0.5, features2=None):

    if features2 is None:
        out = F.normalize(x, dim=-1)
        d = out.size()
        batch_size = d[0] // 2
        out = out.view(batch_size, 2, -1).contiguous()
        out_1 = out[:, 0]
        out_2 = out[:, 1]
    else:
        batch_size = x.shape[0]
        out_1 = F.normalize(x, dim=-1)
        out_2 = F.normalize(features2, dim=-1)
        # out_1 = x
        # out_2 = features2

    # neg score
    out = torch.cat([out_1, out_2], dim=0)
    # print("temperature is {}".format(t))
    neg = torch.exp(torch.mm(out, out.t().contiguous()) / t)

    mask = get_negative_mask(batch_size).cuda()
    neg = neg.masked_select(mask).view(2 * batch_size, -1)

    # pos score
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / t)
    pos = torch.cat([pos, pos], dim=0)

    # estimator g()
    Ng = neg.sum(dim=-1)

    # contrastive loss

    loss = (- torch.log(pos / (pos + Ng)))

    return loss.mean()


def getStatisticsFromTxt(txtName, num_class=1000):
      statistics = [0 for _ in range(num_class)]
      with open(txtName, 'r') as f:
        lines = f.readlines()
      for line in lines:
            s = re.search(r" ([0-9]+)$", line)
            if s is not None:
              statistics[int(s[1])] += 1
      return statistics


def gather_tensor(tensor, local_rank, world_size):
    # gather features
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, tensor)
    tensor_list[local_rank] = tensor
    tensors = torch.cat(tensor_list)
    return tensors

