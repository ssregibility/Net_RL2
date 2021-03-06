# Learning Deeply Shared Filter Bases for Efficient ConvNets

This work proposes a recursive parameter-sharing structure and training mechanism for ConvNets.
- In the proposed ConvNet architecture, convolution layers are decomposed into a filter basis, that can be shared recursively, and layer-specific parts.
- We conjecture that a shared filter basis combined with a small amount of layer-specific parameters can retain, or further enhance, the representation power of individual layers, if a proper training method is applied. 

## Requirements

We conducted experiments under
- python 3.6.9
- pytorch 1.5, torchvision 0.4, cuda10

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
python3 train_ilsvrc.py --lr=0.1 --momentum=0.9 --weight_decay=1e-4 --lambdaR=10 --shared_rank=32 --unique_rank=1 --batch_size=512 --dataset_path=<path_to_dataset> --model=ResNet34_DoubleShared --visible_device=0,1,2,3
```
To train MobileNetV2_SharedDouble model in the paper on ILSVRC2012, run this command:

```train
python3 train_ilsvrc.py --lr=0.1 --momentum=0.9 --weight_decay=1e-5 --lambdaR=10 --batch_size=512 --dataset_path=<path_to_dataset> --model=MobileNetV2_SharedDouble --visible_device=0,1,2,3
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
python3 eval_ilsvrc.py --pretrained=<path_to_model> --model=ResNet34_DoubleShared --shared_rank=32 --unique_rank=1 --dataset_path=<path_to_dataset> --visible_device=0,1,2,3
```

To evaluate proposed MobileNetV2_SharedDouble model on ILSVRC2012, run:

```eval
python3 eval_ilsvrc.py --pretrained=<path_to_model> --model=MobileNetV2_SharedDouble --dataset_path=<path_to_dataset> --visible_device=0,1,2,3
```

Notes
-  ```---model```, ```---shared_rank``` and ```---unique_rank``` options need to be adjusted properly for evaluating pretrained models.


## Results
Testing errors vs. the number of parameters and FLOPs on CIFAR-100. The number of shared basis components (s), and non-shared basis components (u) are varied. Using more shared basis components results in better performance. In contrast, using more non-shared components does not always improve performance.
![Image](images/graph.png?raw=true)


Our model achieves the following performance on :

### CIFAR-100 Classifcation

| Model name         | Top 1 Error  | Params | FLOPs |  |
| ------------------ |---------------- | ------------ | ----- |----|
| ResNet34-S32U1     |     21.79%         |      7.73M     |  1.55G  | [Download](https://drive.google.com/file/d/1DsYNhRBeqAkDGRZGa7oRXHjC0NU9M_f_/view?usp=sharing) |
| DenseNet121-S64U4  |     22.15%         |      5.08M     |  1.43G  | [Download](https://drive.google.com/file/d/1FeLPbEAkkrT2bZTvnCrNOU3dkJxrp3-D/view?usp=sharing) |
| ResNeXt50-S16U1    |     20.09%         |      19.3M     |  2.38G  | [Download](https://drive.google.com/file/d/1e7UlAOFqN0sMwA4jy6tvAzZsvKYpXA6e/view?usp=sharing) |
| MobileNetV2-SharedDouble |     27.54%         |      1.90M     |  0.14G  | [Download](https://drive.google.com/file/d/1V1cPxGyCGO0Fx-W2r9Wpfy81ByKypFuI/view?usp=sharing) |

### CIFAR-10 Classifcation

| Model name         | Top 1 Error  | Params | FLOPs |   |
| ------------------ |---------------- | ------------ | ----- | ----- |
| ResNet32-S16U1\*    |     6.93%         |      0.24M     |  0.16G  | [Download](https://drive.google.com/file/d/1lGqQJEjMVr-ruMV61byFAPJVFJwU2xEU/view?usp=sharing) |
| ResNet56-S16U1     |     7.46%         |      0.22M     |  0.30G | [Download](https://drive.google.com/file/d/1QBdflDIqV254P1sKEeLSVGjnHpn-KPGi/view?usp=sharing) |
| ResNet56-S16U1\*    |     6.30%         |      0.31M     |  0.30G  | [Download](https://drive.google.com/file/d/1CMtt0vOWQcJXKJ98zfRx00XjOEiZEFAD/view?usp=sharing) |

### ILSVRC2012 Classifcation

| Model name         | Top 1 Error  | Top 5 Error | Params | FLOPs |  |
| ------------------ |---------------- | -------------- | ------------ | ----- | ----- |
| ResNet34-S32U1     |     27.83%         |      9.42%       |      8.20M     |  4.98G  | [Download](https://drive.google.com/file/d/1mGZ5-m4x69MjEC-ldraoOBPHtp-TPVRX/view?usp=sharing) |
| ResNet34-S48U1\*     |     26.67%         |      8.54%       |      11.79M     |  6.52G  | [Download](https://drive.google.com/file/d/12pN0JobnfgKKFX0MFNmFJ22BHTojpwIM/view?usp=sharing) |
| ResNet50-Shared     |     23.95%         |      7.14%       |      18.26M     |  8.22G  | [Download](https://drive.google.com/file/d/16XIdAqjqkePCw-Ppf3z2yPUbKNFhXbGl/view?usp=sharing) |
| MobileNetV2-Shared    |     27.61%         |      9.34%       |      3.24M     |  0.66G  | [Download](https://drive.google.com/file/d/1EWYOVj0URjc7j93ciYaRONorlPU2v4DX/view?usp=sharing) |
| MobileNetV2-SharedDouble    |     28.21%         |      9.85%       |      2.98M     |  0.66G  | [Download](https://drive.google.com/file/d/15AF7I8RGRuTHAzZuKpo7cakIpaOU49cw/view?usp=sharing) |




Notes
- ResNet*XX*-S*s*U*u* denotes our model based on original ResNet*XX* with *s* rank of shared filter basis and *u* rank of unique filter basis. Use ```--shared_rank=s``` and ```--unique_rank=u``` for evaluating pretrained models.
- \* denotes having 2 shared bases in each residual block group. Use ```--model=ResNetXX-DoubleShared``` for evaluating these models. Otherwise use ```--model=ResNetXX-SingleShared```.

## Contributing

There is no way to contribute to the code yet - however, this is subject to be changed.
