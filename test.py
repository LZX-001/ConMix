import argparse
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler

import torch_clustering
from Accuracy import cluster_accuracy
from ConMix import train_ConMix
from data.STL10 import CustomSTL10
from models.resnet import resnet18
from models.resnet_prune import prune_resnet18
from models.resnet_prune_multibn import prune_resnet18_dual
from utils import *
import torchvision.transforms as transforms
import torch.distributed as dist
import numpy as np
import copy

from data.cifar10 import CustomCIFAR10
from data.cifar100 import CustomCIFAR100
from data.augmentation import GaussianBlur

from prune.prune_simCLR import train_prune, train_noprune
from prune.mask import Mask, save_mask

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('experiment', type=str)
parser.add_argument('--save-dir', default='save_dir', type=str, help='path to save checkpoint')
parser.add_argument('--data', type=str, default='dataset_dir', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar', help='dataset, [stl, cifar, cifar20]')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epochs', default=1000, type=int, help='number of total epochs to run')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
parser.add_argument('--save_freq', default=50, type=int, help='save frequency /epoch')
parser.add_argument('--checkpoint', default='model_1000.pt', type=str, help='saving pretrained model')
parser.add_argument('--resume', default=True, type=bool, help='if resume training')
parser.add_argument('--optimizer', default='sgd', type=str, help='optimizer type')
parser.add_argument('--lr', default=0.5, type=float, help='optimizer lr')  # 0.5
parser.add_argument('--scheduler', default='cosine', type=str, help='lr scheduler type')
parser.add_argument('--model', default='res18', type=str, help='model type')
parser.add_argument('--temperature', default=0.2, type=float, help='nt_xent temperature')  # 0.2
parser.add_argument('--output_ch', default=128, type=int, help='proj head output feature number')
parser.add_argument('--tag_num', default=100, type=int, help='tag number for mixup')
parser.add_argument('--class_num', default=10, type=int, help='class number of the dataset')
parser.add_argument('--max_num', default=5000, type=int, help='max number of the headest class')
parser.add_argument('--imb_ratio', default=10, type=int, help='imbalance ratio')
parser.add_argument('--warmup_SDCLR', default=200, type=int, help='warm up epochs for hc')
parser.add_argument('--warmup_epoch', default=10, type=int, help='warm up epochs for hc')  # 10
parser.add_argument('--seed', type=int, default=10, help='random seed')
parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--strength', default=1.0, type=float, help='cifar augmentation, color jitter strength')
parser.add_argument('--resizeLower', default=0.1, type=float, help='resize smallest size')

parser.add_argument('--testContrastiveAcc', action='store_true', help="test contrastive acc")
parser.add_argument('--testContrastiveAccTest', action='store_true', help="test contrastive acc in test set")

# contrast with pruned model
parser.add_argument('--prune', action='store_false', help="if contrasting with pruned model")
parser.add_argument('--prune_percent', type=float, default=0.9, help="whole prune percentage")  # 0.9
parser.add_argument('--random_prune_percent', type=float, default=0, help="random prune percentage")
parser.add_argument('--prune_dual_bn', action='store_false', help="if employing dual bn during pruning")

# save mask
parser.add_argument('--mask_save_freq', type=int, default=100, help="freq for saving mask")


def cosine_annealing(step, total_steps, lr_max, lr_min, warmup_steps=0):
    assert warmup_steps >= 0

    if step < warmup_steps:
        lr = lr_max * step / warmup_steps
    else:
        lr = lr_min + (lr_max - lr_min) * 0.5 * (
                1 + np.cos((step - warmup_steps) / (total_steps - warmup_steps) * np.pi))

    return lr


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    global args
    args = parser.parse_args()

    save_dir = os.path.join(args.save_dir, args.experiment)
    if os.path.exists(save_dir) is not True:
        os.system("mkdir -p {}".format(save_dir))

    print("distributing")

    dist.init_process_group(backend="nccl", init_method="env://")
    print("paired")
    torch.cuda.set_device(args.local_rank)

    rank = torch.distributed.get_rank()
    logName = "log.txt"

    log = logger(path=save_dir, local_rank=rank, log_name=logName)
    log.info(str(args))

    setup_seed(args.seed + rank)

    world_size = torch.distributed.get_world_size()
    print("employ {} gpus in total".format(world_size))
    print("rank is {}, world size is {}".format(rank, world_size))

    assert args.batch_size % world_size == 0
    batch_size = args.batch_size // world_size

    # define model
    if args.dataset == 'stl':
        imagenet = True
    elif args.dataset == 'cifar' or args.dataset == 'cifar20':
        imagenet = False
    else:
        assert False

    if args.dataset == 'stl':
        num_class = 10
    elif args.dataset == 'cifar':
        num_class = 10
    elif args.dataset == 'cifar20':
        num_class = 20
    else:
        assert False

    if args.model == 'res18':
        model = resnet18(pretrained=False, imagenet=imagenet, num_classes=num_class)
        if args.prune:
            model = prune_resnet18(pretrained=False, imagenet=imagenet, num_classes=num_class)
            if args.prune_dual_bn:
                model = prune_resnet18_dual(pretrained=False, imagenet=imagenet, num_classes=num_class)

    if not args.prune:
        assert not args.prune_dual_bn

    if model.fc is None:
        # hard coding here, for ride resent
        ch = 192
    else:
        ch = model.fc.in_features

    if args.prune_dual_bn:
        from models.resnet_prune_multibn import proj_head
        model.fc = proj_head(ch, args.output_ch)
    else:
        from models.utils import proj_head
        model.fc = proj_head(ch, args.output_ch)

    model.cuda()

    process_group = torch.distributed.new_group(list(range(world_size)))
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)

    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank, find_unused_parameters=True)

    cudnn.benchmark = True

    if args.dataset == "cifar" or args.dataset == "cifar20":
        rnd_color_jitter = transforms.RandomApply([transforms.ColorJitter(0.4 * args.strength, 0.4 * args.strength,
                                                                          0.4 * args.strength, 0.1 * args.strength)],
                                                  p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)

        tfs_train = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(args.resizeLower, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            rnd_color_jitter,
            rnd_gray,
            transforms.ToTensor(),
        ])

        tfs_test = transforms.Compose([
            transforms.ToTensor(),
        ])
    elif args.dataset == "stl":
        rnd_color_jitter = transforms.RandomApply([transforms.ColorJitter(0.4 * args.strength, 0.4 * args.strength,
                                                                          0.4 * args.strength, 0.1 * args.strength)],
                                                  p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)

        tfs_train = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=(args.resizeLower, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            rnd_color_jitter,
            rnd_gray,
            transforms.ToTensor(),
            # normalize,
        ])
        tfs_test = transforms.Compose([
            transforms.ToTensor(),
        ])
    # dataset process
    if args.dataset == "cifar":
        # the data distribution
        if args.data == '':
            root = '../../data'
        else:
            root = args.data
        train_datasets = CustomCIFAR10(class_num=args.class_num, max_num=args.max_num, imb_ratio=args.imb_ratio,
                                       root=root, train=True, transform=tfs_test, download=True)
    elif args.dataset == "cifar20":
        assert not args.testContrastiveAccTest
        # the data distribution
        if args.data == '':
            root = '../../data'
        else:
            root = args.data
        train_datasets = CustomCIFAR100(class_num=args.class_num, max_num=args.max_num, imb_ratio=args.imb_ratio,
                                        root=root, train=True, transform=tfs_test, download=True)
    elif args.dataset == "stl":
        if args.data == '':
            root = '../../data'
        else:
            root = args.data
        train_datasets = CustomSTL10(class_num=args.class_num, max_num=args.max_num, imb_ratio=args.imb_ratio,
                                     root=root, split='train', transform=tfs_test, download=True)
    else:
        assert False

    shuffle = True
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_datasets, shuffle=shuffle)
    train_loader = torch.utils.data.DataLoader(
        train_datasets,
        num_workers=args.num_workers,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=False)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4, momentum=0.9)
    else:
        print("no defined optimizer")
        assert False

    if args.scheduler == 'constant':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs * len(train_loader) * 10, ],
                                                         gamma=1)
    elif args.scheduler == 'cosine':
        training_iters = args.epochs * len(train_loader)
        warm_up_iters = args.warmup_epoch * len(train_loader)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(step,
                                                    training_iters,
                                                    1,  # since lr_lambda computes multiplicative factor
                                                    1e-6 / args.lr,
                                                    warmup_steps=warm_up_iters)
        )
    else:
        print("unknown schduler: {}".format(args.scheduler))
        assert False

    start_epoch = 1
    if args.checkpoint != '':
        checkpoint = torch.load(os.path.join(save_dir, args.checkpoint), map_location="cuda")

        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

        if 'epoch' in checkpoint and 'optim' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            optimizer.load_state_dict(checkpoint['optim'])

            for i in range((start_epoch - 1) * len(train_loader)):
                scheduler.step()
            log.info("resume the checkpoint {} from epoch {}".format(args.checkpoint, checkpoint['epoch']))
        else:
            log.info("cannot resume since lack of files")
            assert False

    if args.resume:
        if args.checkpoint == '':
            checkpoint = torch.load(os.path.join(save_dir, 'model.pt'), map_location="cuda")
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)

            if 'epoch' in checkpoint and 'optim' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                optimizer.load_state_dict(checkpoint['optim'])

                for i in range((start_epoch - 1) * len(train_loader)):
                    scheduler.step()
                log.info("resume the checkpoint {} from epoch {}".format(args.checkpoint, checkpoint['epoch']))
            else:
                log.info("cannot resume since lack of files")
                assert False
    test_times(model, train_loader, times=10, class_num=args.class_num)


def save_checkpoint(state, filename='weight.pt'):
    """
    Save the training model
    """
    torch.save(state, filename)


def getEmbedding(model, data_loader):
    flag = False
    model.eval()
    for step, (x, y, z) in enumerate(data_loader):
        x = x.to('cuda')
        y = y.to('cuda')
        inputs_1 = x[:, 0, ...]
        inputs_2 = x[:, 1, ...]
        with torch.no_grad():
            model.module.set_prune_flag(False)
            c = model(inputs_1)
            if flag:
                embedding = torch.cat((embedding, c), dim=0)
                target = torch.cat((target, y), dim=0)
            else:
                embedding = c
                target = y
                flag = True
    embedding = torch.nn.functional.normalize(embedding, dim=-1)

    # print(embedding.shape)
    return embedding, target


def clustering(features, n_clusters, random_state=1000):
    kwargs = {
        'metric': 'cosine',
        'distributed': True,
        'random_state': random_state,
        'n_clusters': n_clusters,
        'verbose': False
    }
    clustering_model = torch_clustering.PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)

    psedo_labels = clustering_model.fit_predict(features)
    cluster_centers = clustering_model.cluster_centers_
    return psedo_labels, cluster_centers


def test_times(model, data_loader, times, class_num):
    embedding, target = getEmbedding(model, data_loader)
    random.seed(10)
    random_numbers = [random.randint(0, 10000) for _ in range(times)]
    ACC = []
    NMI = []
    KAPPA = []
    ARI = []
    CA = []
    # METRIC = []
    target = np.asarray(target.cpu())
    cnt = 0
    for i in random_numbers:
        print(cnt)
        cnt += 1
        y_pred, _ = clustering(embedding, class_num, random_state=i)
        y_pred = np.asarray(y_pred.cpu())
        acc, nmi, kappa, ari, ca = cluster_accuracy(target, y_pred)
        print(acc, nmi, kappa, ari, np.mean(ca))
        ACC.append(acc)
        NMI.append(nmi)
        KAPPA.append(kappa)
        ARI.append(ari)
        CA.append(np.mean(ca))
        round(np.mean(ACC), 4)
    print([round(np.mean(ACC), 4), round(np.mean(NMI), 4), round(np.mean(KAPPA), 4), round(np.mean(ARI), 4),
           round(np.mean(CA), 4)])


if __name__ == '__main__':
    main()
