import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import copy

# BasicBlock for single basis models
class BasicBlock_SingleShared(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, unique_rank, shared_basis, stride=1):
        super(BasicBlock_SingleShared, self).__init__()
        
        self.unique_rank = unique_rank
        self.shared_basis = shared_basis
        
        self.total_rank = unique_rank+shared_basis.weight.shape[0]
        
        self.basis_conv1 = nn.Conv2d(in_planes, unique_rank, kernel_size=3, stride=stride, padding=1, bias=False)
        self.basis_bn1 = nn.BatchNorm2d(self.total_rank)
        self.coeff_conv1 = nn.Conv2d(self.total_rank, planes, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.basis_conv2 = nn.Conv2d(planes, unique_rank, kernel_size=3, stride=1, padding=1, bias=False)
        self.basis_bn2 = nn.BatchNorm2d(self.total_rank)
        self.coeff_conv2 = nn.Conv2d(self.total_rank, planes, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.cat((self.basis_conv1(x), self.shared_basis(x)),dim=1)
        out = self.basis_bn1(out)
        out = self.coeff_conv1(out)
        
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        out = torch.cat((self.basis_conv2(out), self.shared_basis(out)),dim=1)
        out = self.basis_bn2(out)
        out = self.coeff_conv2(out)
        
        out = self.bn2(out)

        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        
        return out

# BaiscBlock for double basis models
class BasicBlock_DoubleShared(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, unique_rank, shared_basis_1, shared_basis_2, stride=1):
        super(BasicBlock_DoubleShared, self).__init__()
        
        self.unique_rank = unique_rank
        self.shared_basis_1 = shared_basis_1
        self.shared_basis_2 = shared_basis_2
        
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
        out = torch.cat((self.basis_conv1(x), self.shared_basis_1(x)),dim=1)
        out = self.basis_bn1(out)
        out = self.coeff_conv1(out)
        
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        out = torch.cat((self.basis_conv2(out), self.shared_basis_2(out)),dim=1)
        out = self.basis_bn2(out)
        out = self.coeff_conv2(out)
        
        out = self.bn2(out)

        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        
        return out

# original Bottleneck block    
class BottleneckBlock_Shared(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, shared_basis_1, shared_basis_2, stride=1):
        super(BottleneckBlock_Shared, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.shared_basis_1 = shared_basis_1
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
        #                        stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.conv3 = nn.Conv2d(planes, self.expansion *
        #                        planes, kernel_size=1, bias=False)
        self.shared_basis_2 = shared_basis_2
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn2(self.shared_basis_1(out)))
        # out = self.bn3(self.conv3(out))
        out = self.bn3(self.shared_basis_2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# original Bottleneck block    
class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        #print(out.shape)
        return out


# Original BasicBlock
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        
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
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        
        return out

# Proposed ResNet sharing bottleneck as a basis for each residual block group
class ResNet_BottleneckShared(nn.Module):
    def __init__(self, block_basis, block_original, num_blocks, num_classes=1000):
        super(ResNet_BottleneckShared, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.shared_basis_1_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.shared_basis_1_2 = nn.Conv2d(64, 64*4, kernel_size=1, bias=False)
        self.layer1 = self._make_layer(block_basis, block_original, 64, num_blocks[0], self.shared_basis_1_1, self.shared_basis_1_2, stride=1)
        
        self.shared_basis_2_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.shared_basis_2_2 = nn.Conv2d(128, 128*4, kernel_size=1, bias=False)
        self.layer2 = self._make_layer(block_basis, block_original, 128, num_blocks[1], self.shared_basis_2_1, self.shared_basis_2_2, stride=2)
        
        self.shared_basis_3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.shared_basis_3_2 = nn.Conv2d(256, 256*4, kernel_size=1, bias=False)
        self.layer3 = self._make_layer(block_basis, block_original, 256, num_blocks[2], self.shared_basis_3_1, self.shared_basis_3_2, stride=2)
        
        self.shared_basis_4_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.shared_basis_4_2 = nn.Conv2d(512, 512*4, kernel_size=1, bias=False)
        self.layer4 = self._make_layer(block_basis, block_original, 512, num_blocks[3], self.shared_basis_4_1, self.shared_basis_4_2,stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block_original.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #initialize every con2d first, then initialize shared basis again later
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Each share basis is orthogonal-initialized separately
        torch.nn.init.orthogonal_(self.shared_basis_1_1.weight)
        torch.nn.init.orthogonal_(self.shared_basis_2_1.weight)
        torch.nn.init.orthogonal_(self.shared_basis_3_1.weight)
        torch.nn.init.orthogonal_(self.shared_basis_4_1.weight)

        torch.nn.init.orthogonal_(self.shared_basis_1_2.weight)
        torch.nn.init.orthogonal_(self.shared_basis_2_2.weight)
        torch.nn.init.orthogonal_(self.shared_basis_3_2.weight)
        torch.nn.init.orthogonal_(self.shared_basis_4_2.weight)



    def _make_layer(self, block_basis, block_original, planes, blocks, shared_basis_1, shared_basis_2, stride=1):
        layers = []
        
        layers.append(block_original(self.in_planes, planes, stride))
        
        self.in_planes = planes * block_original.expansion
        for _ in range(1, blocks):
            layers.append(block_basis(self.in_planes, planes, shared_basis_1, shared_basis_2))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out,inplace=True)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
     
        return out


# Proposed ResNet sharing a single basis for each residual block group
class ResNet_SingleShared(nn.Module):
    def __init__(self, block_basis, block_original, num_blocks, shared_rank, unique_rank, num_classes=1000):
        super(ResNet_SingleShared, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.shared_basis_1 = nn.Conv2d(64, shared_rank*1, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block_basis, block_original, 64, num_blocks[0], unique_rank*1, self.shared_basis_1, stride=1)
        
        self.shared_basis_2 = nn.Conv2d(128, shared_rank*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2 = self._make_layer(block_basis, block_original, 128, num_blocks[1], unique_rank*2, self.shared_basis_2, stride=2)
        
        self.shared_basis_3 = nn.Conv2d(256, shared_rank*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3 = self._make_layer(block_basis, block_original, 256, num_blocks[2], unique_rank*4, self.shared_basis_3, stride=2)
        
        self.shared_basis_4 = nn.Conv2d(512, shared_rank*8, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4 = self._make_layer(block_basis, block_original, 512, num_blocks[3], unique_rank*8, self.shared_basis_4, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #initialize every con2d first, then initialize shared basis again later
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Each share basis is orthogonal-initialized separately
        torch.nn.init.orthogonal_(self.shared_basis_1.weight)
        torch.nn.init.orthogonal_(self.shared_basis_2.weight)
        torch.nn.init.orthogonal_(self.shared_basis_3.weight)
        torch.nn.init.orthogonal_(self.shared_basis_4.weight)

    def _make_layer(self, block_basis, block_original, planes, blocks, unique_rank, shared_basis, stride=1):
        layers = []
        
        layers.append(block_original(self.in_planes, planes, stride))
        
        self.in_planes = planes * block_original.expansion
        for _ in range(1, blocks):
            layers.append(block_basis(self.in_planes, planes, unique_rank, shared_basis))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out,inplace=True)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
     
        return out
    
# Proposed ResNet shaing 2 bases for each residual block group
class ResNet_DoubleShared(nn.Module):
    def __init__(self, block_basis, block_original, num_blocks, shared_rank, unique_rank, num_classes=1000):
        super(ResNet_DoubleShared, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.shared_basis_1_1 = nn.Conv2d(64, shared_rank*1, kernel_size=3, stride=1, padding=1, bias=False)
        self.shared_basis_1_2 = nn.Conv2d(64, shared_rank*1, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block_basis, block_original, 64, num_blocks[0], unique_rank*1, self.shared_basis_1_1, self.shared_basis_1_2, stride=1)
        
        self.shared_basis_2_1 = nn.Conv2d(128, shared_rank*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.shared_basis_2_2 = nn.Conv2d(128, shared_rank*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2 = self._make_layer(block_basis, block_original, 128, num_blocks[1], unique_rank*2, self.shared_basis_2_1, self.shared_basis_2_2, stride=2)
        
        self.shared_basis_3_1 = nn.Conv2d(256, shared_rank*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.shared_basis_3_2 = nn.Conv2d(256, shared_rank*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3 = self._make_layer(block_basis, block_original, 256, num_blocks[2], unique_rank*4, self.shared_basis_3_1, self.shared_basis_3_2, stride=2)
        
        self.shared_basis_4_1 = nn.Conv2d(512, shared_rank*8, kernel_size=3, stride=1, padding=1, bias=False)
        self.shared_basis_4_2 = nn.Conv2d(512, shared_rank*8, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4 = self._make_layer(block_basis, block_original, 512, num_blocks[3], unique_rank*8, self.shared_basis_4_1, self.shared_basis_4_2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #initialize every con2d first, then initialize shared basis again later
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Two shared bases are orthogonal-initialized together. 
        for i in range(1,len(num_blocks)+1):
            basis_1 = getattr(self, "shared_basis_"+str(i)+"_1")
            basis_2 = getattr(self, "shared_basis_"+str(i)+"_2")
            out_chan = basis_1.weight.shape[0]
            in_chan =  basis_1.weight.shape[1]
            X = torch.empty(out_chan*2, in_chan, 3, 3)
            torch.nn.init.orthogonal_(X)
            basis_1.weight.data = copy.deepcopy(X[:out_chan,:])
            basis_2.weight.data = copy.deepcopy(X[out_chan:,:])

    def _make_layer(self, block_basis, block_original, planes, blocks, unique_rank, shared_basis_1, shared_basis_2, stride=1):
        layers = []
        
        layers.append(block_original(self.in_planes, planes, stride))
        
        self.in_planes = planes * block_original.expansion
        for _ in range(1, blocks):
            layers.append(block_basis(self.in_planes, planes, unique_rank, shared_basis_1, shared_basis_2))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out,inplace=True)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
     
        return out

# Original ResNet
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out,inplace=True)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
     
        return out

# Original ResNet
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])
# Original ResNet
def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

# Original ResNet
def ResNet50():
    return ResNet(BottleneckBlock, [3, 4, 6, 3])

# A model with 2 shared bases in each residual block group.
def ResNet34_DoubleShared(shared_rank, unique_rank):
    return ResNet_DoubleShared(BasicBlock_DoubleShared, BasicBlock, [3, 4, 6, 3], shared_rank, unique_rank)

# A model with a shared basis in each residual block group.
def ResNet34_SingleShared(shared_rank, unique_rank):
    return ResNet_SingleShared(BasicBlock_SingleShared, BasicBlock, [3, 4, 6, 3], shared_rank, unique_rank)

# A model with a shared basis in each residual block group.
def ResNet50_Shared():
    return ResNet_BottleneckShared(BottleneckBlock_Shared, BottleneckBlock, [3, 4, 6, 3])

def test():
    #net = ResNet50()
    net = ResNet50_Shared()
    #x = torch.randn(256,3,32,32)
    x = torch.randn(16,3,224,224)
    y = net(x)
    print(y.size())
    #print(net)

test()
