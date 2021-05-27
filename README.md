# Deeply Shared Filter Bases for Parameter-Efficient Convolutional Neural Networks

This repository is the official implementation of "Deeply Shared Filter Bases for Parameter-Efficient Convolutional Neural Networks" under submission.

In this paper, we present a recursive convolution block design and training method, in which a recursively shareable part, or a filter basis, 
is separated and learned while effectively avoiding the vanishing/exploding gradients problem during training.

## Requirements

We conducted experiments under
- python 3.8
- pytorch 1.8, torchvision 0.9, cuda11

To install requirements:

```setup
pip3 install -r requirements.txt
```

## Training

To train ResNet34-S48U1 in the paper on ILSVRC2012, run this command:

```train
python3 train_ilsvrc.py --lr=0.1 --momentum=0.9 --weight_decay=1e-4 --lambdaR=10 --shared_rank=48 --unique_rank=1 --batch_size=256 --dataset_path=<path_to_dataset> --model=ResNet34_DoubleShared --visible_device=0,1,2,3
```

To train ResNet50/101-Shared in the paper on ILSVRC2012, run this command:

```train
python3 train_ilsvrc.py --lr=0.1 --momentum=0.9 --weight_decay=1e-4 --lambdaR=10 --batch_size=256 --dataset_path=<path_to_dataset> --model=ResNet50_SharedSingle --visible_device=0,1,2,3
```
To train ResNet50-Shared++ in the paper on ILSVRC2012, run this command:

```train
python3 train_ilsvrc.py --lr=0.1 --momentum=0.9 --weight_decay=1e-4 --lambdaR=10 --batch_size=256 --dataset_path=<path_to_dataset> --model=ResNet50_Shared --visible_device=0,1,2,3
```

To train MobileNetV2-Shared in the paper on ILSVRC2012, run this command:

```train
python3 train_ilsvrc.py --lr=0.1 --momentum=0.9 --weight_decay=1e-5 --lambdaR=10 --batch_size=256 --dataset_path=<path_to_dataset> --model=MobileNetV2_Shared --visible_device=0,1,2,3
```

To train MobileNetV2-Shared++ in the paper on ILSVRC2012, run this command:

```train
python3 train_ilsvrc.py --lr=0.1 --momentum=0.9 --weight_decay=1e-5 --lambdaR=10 --batch_size=256 --dataset_path=<path_to_dataset> --model=MobileNetV2_SharedDouble visible_device=0,1,2,3
```

## Evaluation

To train ResNet34-S48U1 in the paper on ILSVRC2012, run:

```eval
python3 eval_ilsvrc.py --pretrained=<path_to_model> --model=ResNet34_DoubleShared --shared_rank=48 --unique_rank=1 --dataset_path=<path_to_dataset> --visible_device=0,1,2,3
```
To train ResNet50/101-Shared in the paper on ILSVRC2012, run:

```eval
python3 eval_ilsvrc.py --pretrained=<path_to_model> --model=ResNet50_SharedSingle --dataset_path=<path_to_dataset> --visible_device=0,1,2,3
```

To train ResNet50/101-Shared++ in the paper on ILSVRC2012, run:

```eval
python3 eval_ilsvrc.py --pretrained=<path_to_model> --model=ResNet50_Shared --dataset_path=<path_to_dataset> --visible_device=0,1,2,3
```

To evaluate proposed MobileNetV2_Shared model in the paper on ILSVRC2012, run:

```eval
python3 eval_ilsvrc.py --pretrained=<path_to_model> --model=MobileNetV2_Shared --dataset_path=<path_to_dataset> --visible_device=0,1,2,3
```

To evaluate proposed MobileNetV2_Shared++ model in the paper on ILSVRC2012, run:

```eval
python3 eval_ilsvrc.py --pretrained=<path_to_model> --model=MobileNetV2_Shared --dataset_path=<path_to_dataset> --visible_device=0,1,2,3
```

## Results and Pretrained models

Our model achieves the following performance on :

### ILSVRC2012 Classifcation

| Model name         | Top 1 Error  | Top 5 Error | Params | FLOPs |  |
| ------------------ |---------------- | -------------- | ------------ | ----- | ----- |
| ResNet34-S48U1\*     |     26.67%         |      8.54%       |      11.79M     |  3.26G  | [Download](https://drive.google.com/file/d/12pN0JobnfgKKFX0MFNmFJ22BHTojpwIM/view?usp=sharing) |
| ResNet50-Shared     |     23.64%         |      6.98%       |      20.51M     |  4.11G  | [Download](https://drive.google.com/file/d/16XIdAqjqkePCw-Ppf3z2yPUbKNFhXbGl/view?usp=sharing) |
| ResNet50-Shared++     |     23.95%         |      7.14%       |      18.26M     |  4.11G  | [Download]() |
| ResNet101-Shared     |     22.31%         |      6.47%       |      29.47M     |  7.83G  | [Download]() |
| MobileNetV2-Shared    |     27.61%         |      9.34%       |      3.24M     |  0.33G  | [Download](https://drive.google.com/file/d/1EWYOVj0URjc7j93ciYaRONorlPU2v4DX/view?usp=sharing) |
| MobileNetV2-Shared++    |     28.21%         |      9.85%       |      2.98M     |  0.33G  | [Download](https://drive.google.com/file/d/15AF7I8RGRuTHAzZuKpo7cakIpaOU49cw/view?usp=sharing) |


## Contributing

There is no way to contribute to the code yet - however, this is subject to be changed.
