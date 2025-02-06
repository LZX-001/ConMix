import torch

from utils import nt_xent
from prune.prune_simCLR import gatherFeatures
def train_ConMix(train_loader, model, optimizer, epoch, rank, world_size, args=None,updateLR=False,scheduler=None):
    for i, (inputs, target, idx) in enumerate(train_loader):
        if updateLR:
            scheduler.step()

        inputs = inputs.cuda(non_blocking=True)
        input1 = inputs[:, 0, ...]
        input2 = inputs[:, 1, ...]
        model.train()
        optimizer.zero_grad()
        model.module.set_prune_flag(False)  # special
        N = len(idx)
        random_labels = torch.randint(low=0, high=args.tag_num, size=(N,), dtype=torch.int64)
        with torch.no_grad():
            feature1 = model(input1)
            feature1_no_grad = gatherFeatures(feature1, rank, world_size).detach()
            feature1_no_grad = torch.nn.functional.normalize(feature1_no_grad, dim=-1)
        feature2 = model(input2)
        feature2 = gatherFeatures(feature2, rank, world_size)
        feature2 = torch.nn.functional.normalize(feature2, dim=-1)

        loss = generateLoss(feature1_no_grad, feature2, random_labels, args.temperature,args.tag_num)

        loss = loss * world_size
        loss.backward()
        feature1 = model(input1)
        feature1 = gatherFeatures(feature1, rank, world_size)
        feature1 = torch.nn.functional.normalize(feature1, dim=-1)
        loss = generateLoss(feature1, feature2.detach(),random_labels, args.temperature,args.tag_num)
        loss = loss * world_size
        loss.backward()
        optimizer.step()
        if i == 0:
            print("loss: {} in epoch {}".format(loss.detach().cpu(), epoch))
def generateLoss(feature1,feature2,random_labels,temperature,n_labels):
    n_samples = len(random_labels)
    weight = torch.zeros([n_labels,n_samples]).cuda(non_blocking=True)
    weight[random_labels, torch.arange(n_samples)] = 1
    weight = weight[torch.sum(weight, dim=1) != 0]
    weight = torch.nn.functional.normalize(weight, p=1, dim=1)  # l1 normalization
    mix_feature1 = torch.mm(weight,feature1)
    mix_feature2 = torch.mm(weight,feature2)
    mix_feature1 = torch.nn.functional.normalize(mix_feature1, dim=-1)
    mix_feature2 = torch.nn.functional.normalize(mix_feature2, dim=-1)
    loss = nt_xent(x=mix_feature1, t=temperature, features2=mix_feature2)
    return loss