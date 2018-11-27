# Image-to-image translation with flow-based generative model 

![](https://user-images.githubusercontent.com/7057863/49054094-f0c11280-f1c0-11e8-92f0-107c6642c238.png)

## Requirements

 - Tensorflow (tested with v1.8.0)
 - Horovod (tested with v0.13.8) and (Open)MPI

Run
```
pip install -r requirements.txt
```

To setup (Open)MPI, check instructions on Horovod github [page](https://github.com/uber/horovod).

## Train 

Train with 4 GPUs:
```
mpiexec -n 4 python train.py --problem PROBLEM --image_size 32 --n_level 3 --depth 32 --flow_permutation 2 --flow_coupling 1 --seed 0 --learntop --lr 0.001 --n_bits_x 8 --joint_train --logdir LOGDIR
```

Replace `PROBLEM` with [`mnist` | `edges2shoes`].

## Inference

```
python train.py --problem PROBLEM --image_size 32 --n_level 3 --depth 32 --flow_permutation 2 --flow_coupling 1 --seed 0 --learntop --lr 0.001 --n_bits_x 8 --joint_train --logdir LOGDIR --inference
```

Replace `PROBLEM` with [`mnist` | `edges2shoes`].

After running the command, you will get: 

- `z_A.npy`: latent code of images in domain A from test set.
- `z_B.npy`: latent code of images in domain B from test set.
- `A2B.png`: images in domain B translated from domain A.
- `B2A.png`: images in domain A tanslated from domain B.

in `LOGDIR`

## DEBUG: train with 1 GPU

Run wtih small depth to test
```
CUDA_VISIBLE_DEVICES=0 python train.py --depth 1 --epochs_full_sample 1 --epochs_full_valid 1 --joint_train --problem PROBLEM --logdir LOGDIR
```

## WIP

Train with 4 GPUs with B domain pixel l2 loss:
```
mpiexec -n 4 python train.py --problem PROBLEM --image_size 32 --n_level 3 --depth 32 --flow_permutation 2 --flow_coupling 1 --seed 0 --learntop --lr 0.001 --n_bits_x 8 --joint_train --logdir LOGDIR --B_loss --B_loss_fn l2
```


