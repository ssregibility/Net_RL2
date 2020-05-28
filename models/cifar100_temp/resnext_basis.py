import torch
import torch.nn as nn
import torch.nn.functional as F
    
class BottleNeck_Basis(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, groups, base_width, unique_rank, shared_basis, stride=1):
        super().__init__()
        
        self.unique_rank = unique_rank
        self.shared_basis = shared_basis
        
        self.total_rank = unique_rank+shared_basis.weight.shape[0]
        
        groups = groups
        width = int(out_channels * (base_width / 64.)) * groups
        
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        self.basis_conv2 = nn.Conv2d(width, unique_rank, stride=stride, kernel_size=3, padding=1, groups=int(unique_rank/base_width), bias=False)
        #self.basis_bn2 = nn.BatchNorm2d(self.total_rank)
        #self.coeff_conv2 = nn.Conv2d(self.total_rank, width, kernel_size=1, stride=stride, padding=0, bias=False) #아래와 중복 - 건너뜀
        self.bn2 = nn.BatchNorm2d(self.total_rank)
        self.conv3 = nn.Conv2d(self.total_rank, out_channels * BottleNeck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * BottleNeck.expansion)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)),inplace=True)
        out = F.relu(self.bn2(torch.cat((self.basis_conv2(out),self.shared_basis(out)),dim=1)),inplace=True)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out,inplace=True)

        return out
    
class BottleNeck(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, groups=1, base_width=64, stride=1):
        super().__init__()
        width = int(out_channels * (base_width / 64.)) * groups
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, stride=stride, kernel_size=3, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )
        
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNext_Basis(nn.Module):
    def __init__(self, block, block_without_basis, num_blocks, num_classes, groups, base_width, shared_rank,unique_rank):
        super(ResNext_Basis, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        width = int(64 * (base_width / 64.)) * groups
        self.shared_basis_1 = nn.Conv2d(width, shared_rank, kernel_size=3, stride=1, padding=1, groups=int(shared_rank/4), bias=False)
        #nn.init.orthogonal_(self.shared_basis_1.weight)
        self.layer1 = self._make_layer(block, block_without_basis, 64, num_blocks[0], groups, base_width, unique_rank, self.shared_basis_1, stride=1)
        
        width = int(128 * (base_width / 64.)) * groups
        self.shared_basis_2 = nn.Conv2d(width, shared_rank*2, kernel_size=3, stride=1, padding=1, groups=int(shared_rank/4), bias=False)
        #nn.init.orthogonal_(self.shared_basis_2.weight)
        self.layer2 = self._make_layer(block, block_without_basis, 128, num_blocks[1], groups, base_width, unique_rank*2, self.shared_basis_2, stride=2)
        
        width = int(256 * (base_width / 64.)) * groups
        self.shared_basis_3 = nn.Conv2d(width, shared_rank*4, kernel_size=3, stride=1, padding=1, groups=int(shared_rank/4), bias=False)
        #nn.init.orthogonal_(self.shared_basis_3.weight)
        self.layer3 = self._make_layer(block, block_without_basis, 256, num_blocks[2], groups, base_width, unique_rank*4, self.shared_basis_3, stride=2)
        
        width = int(512 * (base_width / 64.)) * groups
        self.shared_basis_4 = nn.Conv2d(width, shared_rank*8, kernel_size=3, stride=1, padding=1, groups=int(shared_rank/4), bias=False)
        #nn.init.orthogonal_(self.shared_basis_4.weight)
        self.layer4 = self._make_layer(block, block_without_basis, 512, num_blocks[3], groups, base_width, unique_rank*8, self.shared_basis_4, stride=2)
        
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, block_without_basis, planes, num_blocks, groups, base_width, unique_rank, shared_basis, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        
        #There is no parameter shraing for a first block of the group
        layers.append(block_without_basis(self.in_planes, planes, groups, base_width, stride))
        self.in_planes = planes * block.expansion
            
        #After the first block, parameter shraing happens in every blocks in the group
        for stride in strides[1:]:
            layers.append(block(self.in_planes, planes, groups, base_width, unique_rank, shared_basis, stride))
            self.in_planes = planes * block.expansion
            
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)),inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNext50_32x4d_Basis(num_classes, shared_rank, unique_rank):
    return ResNext_Basis(BottleNeck_Basis, BottleNeck, [3,4,6,3],num_classes,32,4,shared_rank,unique_rank)
