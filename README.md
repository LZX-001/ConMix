The code is for paper: ConMix: Contrastive Mixup at Representation Level for Long-tailed Deep Clustering (ICLR'25)

## Training
```bash
torchrun  --nproc_per_node=1 --master_port 4831 train.py test_cifar_10_imb_10
```
for training on CIFAR-10 with imbalance ratio = 10, you may need to choose a save_dir first in train.py. 
test_cifar_10_imb_10 is the 'experiment' argument where you save your checkpoint.

For other experiments, you can change the augment in train.py as you like.

## Test
```bash
torchrun  --nproc_per_node=1 --master_port 4831 test.py test_cifar_10_imb_10
```
use the specific experiment name for testing.
## Acknowlegement
SDCLR: https://github.com/VITA-Group/SDCLR

torch_clustering: https://github.com/Hzzone/torch_clustering

## Citation
If this code is helpful, you are welcome to cite our paper.

@inproceedings{
li2025conmix,
title={ConMix: Contrastive Mixup at Representation Level for Long-tailed Deep Clustering},
author={Zhixin Li and Yuheng Jia},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=3lH8WT0fhu}
}
