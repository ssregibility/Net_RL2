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
    
class BottleNeck_Basis(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, growth_rate, unique_rank, shared_basis):
        super().__init__()
        
        inner_channels = 4 * growth_rate
        
        self.in_channels = in_channels
        self.shared_basis = shared_basis
        self.total_rank = unique_rank+shared_basis.weight.shape[0]
        
        """
        self.bn0 = nn.BatchNorm2d(in_channels)
        #relu
        self.conv1 = nn.Conv2d(in_channels, unique_rank, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.total_rank)
        #relu
        self.basis_conv2 = nn.Conv2d(inner_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        self.basis_bn2 = nn.BatchNorm2d(inner_channels)
        self.coeff_conv2 = nn.Conv2d(self.total_rank, inner_channels, kernel_size=1, bias=False)
        """
        
        self.bn0 = nn.BatchNorm2d(in_channels)
        #relu
        self.conv1 = nn.Conv2d(in_channels, inner_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inner_channels)
        #relu
        self.basis_conv2 = nn.Conv2d(inner_channels, unique_rank, kernel_size=3, padding=1, bias=False)
        self.basis_bn2 = nn.BatchNorm2d(self.total_rank)
        self.coeff_conv2 = nn.Conv2d(self.total_rank, growth_rate, kernel_size=1, bias=False)

    def forward(self, x):
        out = F.relu(self.bn0(x),inplace=True)
        out = F.relu(self.bn1(self.conv1(out)),inplace=True)
        out = torch.cat((self.basis_conv2(out),self.shared_basis(out)),dim=1)
        out = self.basis_bn2(out)
        out = self.coeff_conv2(out)
        return torch.cat((x, out), 1)

#DesneNet-BC
#B stands for bottleneck layer(BN-RELU-CONV(1x1)-BN-RELU-CONV(3x3))
#C stands for compression factor(0<=theta<=1)
class DenseNet_Basis(nn.Module):
    def __init__(self, block, num_blocks, num_classes, shared_rank, unique_rank, growth_rate=12, reduction=0.5):
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
        
        self.shared_basis_1 = nn.Conv2d(inner_channels*2, shared_rank, kernel_size=3, padding=1, bias=False)
        self.shared_basis_2 = nn.Conv2d(inner_channels*2, shared_rank, kernel_size=3, padding=1, bias=False)
        self.shared_basis_3 = nn.Conv2d(inner_channels*2, shared_rank, kernel_size=3, padding=1, bias=False)
        self.shared_basis_4 = nn.Conv2d(inner_channels*2, shared_rank, kernel_size=3, padding=1, bias=False)
        
        for index in range(len(num_blocks) - 1):
            self.features.add_module("dense_block_layer_{}".format(index), self._make_dense_layers(block, inner_channels, num_blocks[index], unique_rank, getattr(self,"shared_basis_"+str(index+1))))
            inner_channels += growth_rate * num_blocks[index]

            #"""If a dense block contains m feature-maps, we let the 
            #following transition layer generate θm output feature-
            #maps, where 0 < θ ≤ 1 is referred to as the compression 
            #fac-tor.
            out_channels = int(reduction * inner_channels) # int() will automatic floor the value
            self.features.add_module("transition_layer_{}".format(index), Transition(inner_channels, out_channels))
            inner_channels = out_channels

        self.features.add_module("dense_block_layer_{}".format(len(num_blocks) - 1), self._make_dense_layers(block, inner_channels, num_blocks[len(num_blocks)-1], unique_rank, getattr(self,"shared_basis_"+str(index+1))))
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

    def _make_dense_layers(self, block, in_channels, num_blocks, unique_rank, shared_basis):
        dense_block = nn.Sequential()
        for index in range(num_blocks):
            dense_block.add_module('bottle_neck_layer_{}'.format(index), block(in_channels, self.growth_rate, unique_rank, shared_basis))
            in_channels += self.growth_rate
        return dense_block

#224
#480
#982
#992
    
def densenet121_Basis(shared_rank, unique_rank):
    return DenseNet_Basis(BottleNeck_Basis, [6,12,24,16], 100, shared_rank, unique_rank, growth_rate=32)

def densenet169_Basis(shared_rank, unique_rank):
    return DenseNet_Basis(BottleNeck_Basis, [6,12,32,32], 100, shared_rank, unique_rank, growth_rate=32)

def densenet201_Basis(shared_rank, unique_rank):
    return DenseNet_Basis(BottleNeck_Basis, [6,12,48,32], 100, shared_rank, unique_rank, growth_rate=32)