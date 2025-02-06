## Requirements
faiss==1.5.3
matplotlib==3.7.3
munkres==1.1.4
numpy==1.24.4
numpy==1.24.3
Pillow==10.0.1
Pillow==10.3.0
scikit_learn==1.3.0
scipy==1.10.1
torch==1.13.1+cu117
torchvision==0.14.1+cu117
tqdm==4.66.1

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