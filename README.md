# Learning Shared Filter Bases for Efficient ConvNets

This repository is the official implementation of **Learning Shared Filter Bases for Efficient ConvNets**, a NIPS2020 submission.

## Requirements

To install requirements:

```setup
pip3 install -r requirements.txt
```

## Training

To train the models in the paper on CIFAR-10, run this command:

```train
python3 train_cifar10.py --lr=0.1 --momentum=0.9 --weight_decay=5e-4 --lambdaR=10 --shared_rank=16 --unique_rank=1 --batch_size=256 --model=ResNet56_DoubleShared
```

To train the models in the paper on CIFAR-100, run this command:

```train
python3 train_cifar100.py --lr=0.1 --momentum=0.9 --weight_decay=5e-4 --lambdaR=10 --shared_rank=16 --unique_rank=1 --batch_size=256 --model=ResNet34_SingleShared
```

To train the models in the paper on ILSVRC2012, run this command:

```train
python3 train_ilsvrc.py --lr=0.1 --momentum=0.9 --weight_decay=1e-4 --lambdaR=10 --shared_rank=32 --unique_rank=1 --batch_size=256 --dataset_path=<path_to_dataset> --model=ResNet34_DoubleShared
```

## Evaluation

To evaluate proposed models on CIFAR-10, run:

```eval
python3 eval_cifar10.py --pretrained=<path_to_model> --model=<model_to_evaluate> --visible_device=<CUDA_NUM_to_use> --shared_rank=<num_of_shared_base> --unique_rank=<num_of_unique_base> --batch_size=<batch_size>
```

To evaluate proposed models on CIFAR-100, run:

```eval
python3 eval_cifar100.py --pretrained=<path_to_model> --model=<model_to_evaluate> --visible_device=<CUDA_NUM_to_use> --shared_rank=<num_of_shared_base> --unique_rank=<num_of_unique_base> --batch_size=<batch_size>
```

To evaluate proposed models on ILSVRC2012, run:

```eval
python3 eval_ilsvrc.py --pretrained=<path_to_model> --model=<model_to_evaluate> --visible_device=<CUDA_NUM_to_use> --shared_rank=<num_of_shared_base> --unique_rank=<num_of_unique_base> --batch_size=<batch_size> --dataset_path=<path_to_dataset>
```

## Pre-trained Models

Following files are pre-trained models trained using the proposed method:

> CIFAR100

- **[CIFAR100_ResNet34_SingleShared_S8U1_23.11err.pth](https://drive.google.com/file/d/13fPb-RoTwq5h7NqZ_vq5onNU7qfJuFhT/view?usp=sharing)** trained on CIFAR100, ResNet34-S8U1.
- **[CIFAR100_ResNet34_SingleShared_S16U1_22.64err.pth](https://drive.google.com/file/d/1-x4AvZu68ASVfz4lEmH90HXz8gEUvPjN/view?usp=sharing)** trained on CIFAR100, ResNet34-S16U1.
- **[CIFAR100_ResNet34_SingleShared_S32U1_21.79err.pth](https://drive.google.com/file/d/1O0IskfztEklykdFMrfNMVHGJTKJQD6Am/view?usp=sharing)** trained on CIFAR100, ResNet34-S32U1.
- **[C[FAR100_DenseNet121_SingleShared_S16U1_22.15err.pth](https://drive.google.com/file/d/13XyNHV9qRGyACKOnUY1dTf3p211yJgA5/view?usp=sharing)** trained on CIFAR100, DenseNet121-S16U1.
- **[CIFAR100_ResNext50_SingleShared_S64U4_20.09err.pth](https://drive.google.com/file/d/1nLWETVMwZbGXQ8Ta6vtaYI5SuedUcMAm/view?usp=sharing)** trained on CIFAR100, ResNeXt50-S64U4.

> CIFAR10

- **[CIFAR10_ResNet32_DoubleShared_S16U1_6.93err.pth](https://drive.google.com/file/d/1ZB5yZgMUhU9TGruZpInwX9UQo8kZXEHH/view?usp=sharing)** trained on CIFAR10, ResNet32-S16U1\*.
- **[CIFAR10_ResNet56_DoubleShared_S16U1_6.30err.pth](https://drive.google.com/file/d/1zBQTvDYdbqnfdX3NA6mYy0lHvn68ANRl/view?usp=sharing)** trained on CIFAR10, ResNet56-S16U1\*.
- **[CIFAR10_ResNet32_SingleShared_S8U1_8.08err.pth](https://drive.google.com/file/d/1QmKmICZKk6h_FnctIr6LQrtFCCvWtcac/view?usp=sharing)** trained on CIFAR10, ResNet34-S8U1.
- **[CIFAR10_ResNet32_SingleShared_S16U1_7.43err.pth](https://drive.google.com/file/d/1cpCYf6iwN27RIDjmPxPSTXUW3htZ8-P5/view?usp=sharing)** trained on CIFAR10, ResNet34-S16U1.
- **[CIFAR10_ResNet56_SingleShared_S8U1_7.52err.pth.pth](https://drive.google.com/file/d/1wUB3PnZ8lnSqXFTWGEk1eoLseSFQ2-Tj/view?usp=sharing)** trained on CIFAR10, ResNet56-S8U1.
- **[CIFAR10_ResNet56_SingleShared_S16U1_7.46err.pth.pth](https://drive.google.com/file/d/17rwH4_KNGX2nBgF0PBbBeKfve5IudZrY/view?usp=sharing)** trained on CIFAR10, ResNet56-S16U1.

> ILSVRC2012

- **[ILSVRC_ResNet34_DoubleShared_S32U1_27.69err.pth](https://drive.google.com/file/d/1dtq8TaF88ELnIn4fQr4-eyMGwYCiGYVA/view?usp=sharing)** trained on ILSVRC2012, ResNet34-S32U1\*.
- **[ILSVRC_ResNet34_SingleShared_S32U1_28.42err.pth](https://drive.google.com/file/d/1OgodlaaYYdYXgRFGAMxP_039R5JkUAij/view?usp=sharing)** trained on ILSVRC2012, ResNet34-S32U1.
- **[ILSVRC_ResNet34_SingleShared_S48U1_27.88err.pth](https://drive.google.com/file/d/1NHBvlYrTJzuJuKJjIdtlt5krDiXkue2r/view?usp=sharing)** trained on ILSVRC2012, ResNet34-S48U1.


## Results

Following graphs are comparison between original [ResNets](https://arxiv.org/abs/1512.03385) and ResNet34s using the proposed method :
![Image](https://github.com/ssregibility/Net_RL2/blob/master/images/graph.png?raw=true)


Our model achieves the following performance on :

### CIFAR-100 Classifcation

| Model name         | Top 1 Error  | Params | FLOPs |
| ------------------ |---------------- | ------------ | ----- |
| ResNet34-S8U1      |     23.11%         |      5.87M     |  0.79G  |
| ResNet34-S16U1     |     22.64%         |      6.49M     |  1.05G  |
| ResNet34-S32U1     |     21.79%         |      7.73M     |  1.55G  |
| DenseNet121-S64U4  |     22.15%         |      5.08M     |  1.43G  |
| ResNeXt50-S16U1    |     20.09%         |      19.3M     |  2.38G  |

### CIFAR-10 Classifcation

| Model name         | Top 1 Error  | Params | FLOPs |
| ------------------ |---------------- | ------------ | ----- |
| ResNet32-S8U1      |     8.08%         |      0.15M     |  0.10G  |
| ResNet32-S16U1     |     7.43%         |      0.20M     |  0.16G  |
| ResNet56-S8U1      |     7.52%         |      0.20M     |  0.17G  |
| ResNet56-S16U1     |     7.46%         |      0.22M     |  0.30G |
| ResNet32-S16U1\*    |     6.93%         |      0.24M     |  0.30G  |
| ResNet56-S16U1\*    |     6.30%         |      0.31M     |  0.30G  |

\* denotes having 2 shared bases in each residual block group.

### ILSVRC2012 Classifcation

| Model name         | Top 1 Error  | Top 5 Error | Params | FLOPs |
| ------------------ |---------------- | -------------- | ------------ | ----- |
| ResNet34-S32U1     |     28.42%         |      9.55%       |      8.20M     |  4.98G  |
| ResNet34-S48U1     |     27.88%         |      9.29%       |      9.44M     |  6.52G  |
| ResNet34-S32U1\*    |     27.69%         |      9.11%       |      9.76M     |  4.98G  |

\* denotes having 2 shared bases in each residual block group.

## Contributing

There is no way to contribute to the code yet - however, this is subject to be changed.
