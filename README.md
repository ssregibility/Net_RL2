# Learning Shared Filter Bases for Efficient ConvNets

This repository is the official implementation of Learning Shared Filter Bases for Efficient ConvNets, NIPS2020 submission.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

## Pre-trained Models

Pre-trained models are included in following directory:
./pretrained

> CIFAR100

- **CIFAR100_ResNet34_S8U1_23.11err.pth** trained on CIFAR100, ResNet34-S8U1.
- **CIFAR100_ResNet34_S16U1_22.64err.pth** trained on CIFAR100, ResNet34-S16U1.
- **CIFAR100_ResNet34_S32U1_21.79err.pth** trained on CIFAR100, ResNet34-S32U1.
- **CIFAR100_ResNet34_S16U0_23.43err.pth** trained on CIFAR100, ResNet34-S16U0.
- **CIFAR100_ResNet34_S32U0_22.32err.pth** trained on CIFAR100, ResNet34-S32U0.
- **CIFAR100_DenseNet121_S64U4_22.15err.pth** trained on CIFAR100, DenseNet121-S64U4.

> CIFAR10

- **CIFAR10_ResNet32_Double_S16U1_6.93err.pth** trained on CIFAR10, ResNet32-S16U1\*.
- **CIFAR10_ResNet56_Double_S16U1_6.30err.pth** trained on CIFAR10, ResNet56-S16U1\*.
- **CIFAR10_ResNet34_S8U1_8.08err.pth** trained on CIFAR10, ResNet34-S8U1.
- **CIFAR10_ResNet34_S16U1_7.43err.pth** trained on CIFAR10, ResNet34-S16U1.
- **CIFAR10_ResNet56_S8U1_7.52err.pth.pth** trained on CIFAR10, ResNet56-S8U1.
- **CIFAR10_ResNet56_S16U1_7.46err.pth.pth** trained on CIFAR10, ResNet56-S16U1.

> ILSVRC2012

- **ILSVRC_ResNet34_Double_S32U1_27.69err.pth** trained on ILSVRC2012, ResNet34-S32U1\*.
- **ILSVRC_ResNet34_S32U1_28.42err.pth** trained on ILSVRC2012, ResNet34-S32U1.
- **ILSVRC_ResNet34_S48U1_27.88err.pth** trained on ILSVRC2012, ResNet34-S48U1.


## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

## Contributing
