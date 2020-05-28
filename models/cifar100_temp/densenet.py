import torch
import torch.nn as nn
import torch.nn.functional as F
    
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.down_sample(x)
    
class BottleNeck(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        
        inner_channels = 4 * growth_rate
        
        self.residual_function = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inner_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, growth_rate, kernel_size=3, padding=1, bias=False),
        )
        
    def forward(self, x):
        return torch.cat([x, self.residual_function(x)], 1)

#DesneNet-BC
#B stands for bottleneck layer(BN-RELU-CONV(1x1)-BN-RELU-CONV(3x3))
#C stands for compression factor(0<=theta<=1)
class DenseNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, growth_rate=12, reduction=0.5):
        super().__init__()
        self.growth_rate = growth_rate

        #"""Before entering the first dense block, a convolution 
        #with 16 (or twice the growth rate for DenseNet-BC) 
        #output channels is performed on the input images."""
        inner_channels = 2 * growth_rate

        #For convolutional layers with kernel size 3×3, each 
        #side of the inputs is zero-padded by one pixel to keep 
        #the feature-map size fixed.
        self.conv1 = nn.Conv2d(3, inner_channels, kernel_size=3, padding=1, bias=False) 

        self.features = nn.Sequential()

        for index in range(len(num_blocks) - 1):
            self.features.add_module("dense_block_layer_{}".format(index), self._make_dense_layers(block, inner_channels, num_blocks[index]))
            inner_channels += growth_rate * num_blocks[index]

            #"""If a dense block contains m feature-maps, we let the 
            #following transition layer generate θm output feature-
            #maps, where 0 < θ ≤ 1 is referred to as the compression 
            #fac-tor.
            out_channels = int(reduction * inner_channels) # int() will automatic floor the value
            self.features.add_module("transition_layer_{}".format(index), Transition(inner_channels, out_channels))
            inner_channels = out_channels

        self.features.add_module("dense_block{}".format(len(num_blocks) - 1), self._make_dense_layers(block, inner_channels, num_blocks[len(num_blocks)-1]))
        inner_channels += growth_rate * num_blocks[len(num_blocks) - 1]
        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(inner_channels, num_classes)

    def forward(self, x):
        output = self.conv1(x)
        output = self.features(output)
        output = self.avgpool(output)
        output = output.view(output.size()[0], -1)
        output = self.linear(output)
        return output

    def _make_dense_layers(self, block, in_channels, num_blocks):
        dense_block = nn.Sequential()
        for index in range(num_blocks):
            dense_block.add_module('bottle_neck_layer_{}'.format(index), block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return dense_block


def densenet121(c):
    return DenseNet(BottleNeck, [6,12,24,16], c, growth_rate=32)

def densenet169(c):
    return DenseNet(BottleNeck, [6,12,32,32], c, growth_rate=32)

def densenet201(c):
    return DenseNet(BottleNeck, [6,12,48,32], c, growth_rate=32)

def densenet161(c):
    return DenseNet(BottleNeck, [6,12,36,24], c, growth_rate=48)
