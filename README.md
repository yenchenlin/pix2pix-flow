# Glow

## Requirements

 - Tensorflow (tested with v1.8.0)
 - Horovod (tested with v0.13.8) and (Open)MPI

Run
```
pip install -r requirements.txt
```

To setup (Open)MPI, check instructions on Horovod github [page](https://github.com/uber/horovod).

## DEBUG: train with 1 GPU

Run wtih small depth to test
```
CUDA_VISIBLE_DEVICES=0 python train.py --depth 1 --epochs_full_sample 1 --epochs_full_valid 1 --joint_train --problem PROBLEM --logdir LOGDIR
```

## Train 

Train with 4 GPUs, code_last l2 loss:
```
mpiexec -n 4 python train.py --problem PROBLEM --image_size 32 --n_level 3 --depth 32 --flow_permutation 2 --flow_coupling 1 --seed 0 --learntop --lr 0.001 --n_bits_x 8 --joint_train --logdir LOGDIR
```

Train with 4 GPUs, code_last l2 loss + B domain pixel l2 loss:
```
mpiexec -n 4 python train.py --problem PROBLEM --image_size 32 --n_level 3 --depth 32 --flow_permutation 2 --flow_coupling 1 --seed 0 --learntop --lr 0.001 --n_bits_x 8 --joint_train --logdir LOGDIR --B_loss --B_loss_fn l2
```

## Inference

Add `--inference` to training command, will store `z_A.npy`, `z__B.npy`, `A2B.png`, `B2A.png` in `LOGDIR`.
```
python train.py --problem PROBLEM --image_size 32 --n_level 3 --depth 32 --flow_permutation 2 --flow_coupling 1 --seed 0 --learntop --lr 0.001 --n_bits_x 8 --joint_train --logdir LOGDIR --inference
```

## Datasets

`/afs/csail.mit.edu/u/y/yenchenlin/Workspace/pix2pix/datasets`
