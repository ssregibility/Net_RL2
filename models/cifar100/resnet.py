'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''

#imported from https://github.com/bearpaw/pytorch-classification

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class BasicBlock_Basis(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, unique_rank, shared_basis_1, shared_basis_2, stride=1):
        super(BasicBlock_Basis, self).__init__()
        
        self.unique_rank = unique_rank
        self.shared_basis_1 = shared_basis_1
        self.shared_basis_2 = shared_basis_2
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        
        self.total_rank_1 = unique_rank+shared_basis_1.weight.shape[0]
        self.total_rank_2 = unique_rank+shared_basis_2.weight.shape[0]
        
        self.basis_conv1 = nn.Conv2d(in_planes, unique_rank, kernel_size=3, stride=stride, padding=1, bias=False)
        self.basis_bn1 = nn.BatchNorm2d(self.total_rank_1)
        self.coeff_conv1 = nn.Conv2d(self.total_rank_1, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.basis_conv2 = nn.Conv2d(planes, unique_rank, kernel_size=3, stride=1, padding=1, bias=False)
        self.basis_bn2 = nn.BatchNorm2d(self.total_rank_2)
        self.coeff_conv2 = nn.Conv2d(self.total_rank_2, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        
        if self.stride != 1:
            self.shared_basis_1.stride = (2,2)
        else:
            self.shared_basis_1.stride = (1,1)
                    
        out = torch.cat((self.basis_conv1(x), self.shared_basis_1(x)),dim=1)

        out = self.basis_bn1(out)
        out = self.coeff_conv1(out)
        
        out = self.bn1(out)
        out = self.relu(out)

        self.shared_basis_1.stride = (1,1)

        out = torch.cat((self.basis_conv2(out), self.shared_basis_2(out)),dim=1)
        
        out = self.basis_bn2(out)
        out = self.coeff_conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = out + self.shortcut(x)
        out = self.relu(out)
        
        return out
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)
        
        return out
    
class ResNet_Basis(nn.Module):
    def __init__(self, block_basis, block_original, num_blocks, shared_rank, unique_rank, num_classes=100):
        super(ResNet_Basis, self).__init__()
        self.in_planes = 64
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.shared_basis_1 = nn.Conv2d(64, shared_rank, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block_basis, block_original, 16, num_blocks[0], unique_rank, self.shared_basis_1, self.shared_basis_1, stride=1)
        
        self.shared_basis_2 = nn.Conv2d(128, shared_rank*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2 = self._make_layer(block_basis, block_original, 32, num_blocks[1], unique_rank*2, self.shared_basis_1, self.shared_basis_2, stride=2)
        
        self.shared_basis_3 = nn.Conv2d(256, shared_rank*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3 = self._make_layer(block_basis, block_original, 64, num_blocks[2], unique_rank*4, self.shared_basis_2, self.shared_basis_3, stride=2)
        
        self.shared_basis_4 = nn.Conv2d(512, shared_rank*8, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3 = self._make_layer(block_basis, block_original, 64, num_blocks[2], unique_rank*8, self.shared_basis_3, self.shared_basis_4, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            #if isinstance(m, Bottleneck):
            #    nn.init.constant_(m.bn3.weight, 0)
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block_basis, block_original, planes, blocks, unique_rank, shared_basis_1, shared_basis_2, stride=1):
        layers = []
        #layers.append(block_original(self.in_planes, planes, stride))
        layers.append(block_basis(self.in_planes, planes, unique_rank, shared_basis_1, shared_basis_2, stride=stride))
        self.in_planes = planes * block_original.expansion
        for _ in range(1, blocks):
            layers.append(block_basis(self.in_planes, planes, unique_rank, shared_basis_2, shared_basis_2))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
     
        return x

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[2], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            #if isinstance(m, Bottleneck):
            #    nn.init.constant_(m.bn3.weight, 0)
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
     
        return x

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])
