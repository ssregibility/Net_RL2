# Learning Shared Filter Bases for Efficient ConvNets

This repository is the official implementation of **Learning Shared Filter Bases for Efficient ConvNets**, a NIPS2020 submission.
- We propose to share filter bases of decomposed convolution layers for effective  sharing  of  parameters in ConvNets.
- In overparameterized networks, our method outperforms much deeper counterpart original networks while reducing parameters and computational costs substantially.
![Image](https://github.com/ssregibility/Net_RL2/blob/master/images/conv_decomp.jpg?raw=true)


## Requirements

We conducted experiments under
- python 3.6.9
- pytorch 1.5.0, torchvision 0.5.0, cuda10

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
python3 eval_cifar10.py --pretrained=<path_to_model> --model=ResNet56_DoubleShared --shared_rank=16 --unique_rank=1
```

To evaluate proposed models on CIFAR-100, run:

```eval
python3 eval_cifar100.py --pretrained=<path_to_model> --model=ResNet34_SingleShared --shared_rank=16 --unique_rank=1
```

To evaluate proposed models on ILSVRC2012, run:

```eval
python3 eval_ilsvrc.py --pretrained=<path_to_model> --model=ResNet34_DoubleShared --shared_rank=32 --unique_rank=1 --dataset_path=<path_to_dataset>
```

Notes
-  ```---model```, ```---shared_rank``` and ```---unique_rank``` options need to be properly specified for evaluating pretrained models.


## Results

Following graphs are comparison between original [ResNets](https://arxiv.org/abs/1512.03385) and ResNet34s using the proposed method :
![Image](https://github.com/ssregibility/Net_RL2/blob/master/images/graph.png?raw=true)


Our model achieves the following performance on :

### CIFAR-100 Classifcation

| Model name         | Top 1 Error  | Params | FLOPs |  |
| ------------------ |---------------- | ------------ | ----- |----|
| ResNet34-S8U1      |     23.11%         |      5.87M     |  0.79G  | [Download](https://drive.google.com/file/d/13fPb-RoTwq5h7NqZ_vq5onNU7qfJuFhT/view?usp=sharing) |
| ResNet34-S16U1     |     22.64%         |      6.49M     |  1.05G  | [Download](https://drive.google.com/file/d/1-x4AvZu68ASVfz4lEmH90HXz8gEUvPjN/view?usp=sharing) |
| ResNet34-S32U1     |     21.79%         |      7.73M     |  1.55G  | [Download](https://drive.google.com/file/d/1O0IskfztEklykdFMrfNMVHGJTKJQD6Am/view?usp=sharing) |
| DenseNet121-S64U4  |     22.15%         |      5.08M     |  1.43G  | [Download](https://drive.google.com/file/d/13XyNHV9qRGyACKOnUY1dTf3p211yJgA5/view?usp=sharing) |
| ResNeXt50-S16U1    |     20.09%         |      19.3M     |  2.38G  | [Download](https://drive.google.com/file/d/1nLWETVMwZbGXQ8Ta6vtaYI5SuedUcMAm/view?usp=sharing) |

### CIFAR-10 Classifcation

| Model name         | Top 1 Error  | Params | FLOPs |   |
| ------------------ |---------------- | ------------ | ----- | ----- |
| ResNet32-S8U1      |     8.08%         |      0.15M     |  0.10G  | [Download](https://drive.google.com/file/d/1QmKmICZKk6h_FnctIr6LQrtFCCvWtcac/view?usp=sharing) |
| ResNet32-S16U1     |     7.43%         |      0.20M     |  0.16G  | [Download](https://drive.google.com/file/d/1cpCYf6iwN27RIDjmPxPSTXUW3htZ8-P5/view?usp=sharing) |
| ResNet56-S8U1      |     7.52%         |      0.20M     |  0.17G  | [Download](https://drive.google.com/file/d/1wUB3PnZ8lnSqXFTWGEk1eoLseSFQ2-Tj/view?usp=sharing) |
| ResNet56-S16U1     |     7.46%         |      0.22M     |  0.30G | [Download](https://drive.google.com/file/d/17rwH4_KNGX2nBgF0PBbBeKfve5IudZrY/view?usp=sharing) |
| ResNet32-S16U1\*    |     6.93%         |      0.24M     |  0.30G  | [Download](https://drive.google.com/file/d/1ZB5yZgMUhU9TGruZpInwX9UQo8kZXEHH/view?usp=sharing) |
| ResNet56-S16U1\*    |     6.30%         |      0.31M     |  0.30G  | [Download](https://drive.google.com/file/d/1zBQTvDYdbqnfdX3NA6mYy0lHvn68ANRl/view?usp=sharing) |

\* denotes having 2 shared bases in each residual block group. 
Use ```--model=ResNetXX-DoubleShared``` for these models.

### ILSVRC2012 Classifcation

| Model name         | Top 1 Error  | Top 5 Error | Params | FLOPs |  |
| ------------------ |---------------- | -------------- | ------------ | ----- | ----- |
| ResNet34-S32U1     |     28.42%         |      9.55%       |      8.20M     |  4.98G  | [Download](https://drive.google.com/file/d/1OgodlaaYYdYXgRFGAMxP_039R5JkUAij/view?usp=sharing) |
| ResNet34-S48U1     |     27.88%         |      9.29%       |      9.44M     |  6.52G  | [Download](https://drive.google.com/file/d/1NHBvlYrTJzuJuKJjIdtlt5krDiXkue2r/view?usp=sharing) |
| ResNet34-S32U1\*    |     27.69%         |      9.11%       |      9.76M     |  4.98G  | [Download](https://drive.google.com/file/d/1dtq8TaF88ELnIn4fQr4-eyMGwYCiGYVA/view?usp=sharing) |

Notes
- ResNet*XX*-S*s*U*u* denotes our model based on original ResNet*XX* model with *s* rank of shared filter basis and *u* rank of unique filter basis. Use ```--shared_rank=s``` and ```--unique_rank=u``` for evaluating pretrained models.
- \* denotes having 2 shared bases in each residual block group. Use ```--model=ResNetXX-DoubleShared``` for evaluating these models. Otherwise use ```--model=ResNetXX-SingleShared```.

## Contributing

There is no way to contribute to the code yet - however, this is subject to be changed.
