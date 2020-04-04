import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock_Unique(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, rank, stride=1):
        super(BasicBlock_Unique, self).__init__()
        
        #training mode selection - should be manually changed prior to training
        self.mode = 'train_basis'

        self.rank = rank
        
        self.basis_conv1 = nn.Conv2d(in_planes, rank, kernel_size=3, stride=stride, padding=1, bias=False)
        self.basis_bn1 = nn.BatchNorm2d(self.rank)
        self.coeff_conv1 = nn.Conv2d(self.rank, planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.basis_conv2 = nn.Conv2d(planes, rank, kernel_size=3, stride=stride, padding=1, bias=False)
        self.basis_bn2 = nn.BatchNorm2d(self.rank)
        self.coeff_conv2 = nn.Conv2d(self.rank, planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        #forward pass for basis trainig
        if (self.mode == 'train_basis'):
            out = self.basis_bn1(self.basis_conv1(x))
            out = F.relu(self.bn1(self.coeff_conv1(out)))
            out = self.bn2( self.coeff_conv2( self.basis_bn2(self.basis_conv2(out))))
            out += self.shortcut(x)
            out = F.relu(out)
            return out

#Basic Block with prameter sharing
#in_planes: Integer, number of input channels
#planes: Integer, number of output channels
#rank: Integer, rank of base(=total number of shared templates)
#shared_basis: tensor, tensor for shared base
#stride: Integer, stride

class BasicBlock_Basis(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, rank, shared_basis, stride=1):
        super(BasicBlock_Basis, self).__init__()
        
        #training mode selection - should be manually changed prior to training
        self.mode = 'train_coeffs'
        #self.mode = 'train_basis'
        
        self.shared_basis = shared_basis

        self.rank = rank
        self.total_rank = rank+shared_basis.weight.shape[0]
        
        self.basis_conv1 = nn.Conv2d(in_planes, rank, kernel_size=3, stride=stride, padding=1, bias=False)
        self.basis_bn1 = nn.BatchNorm2d(self.total_rank)
        self.coeff_conv1 = nn.Conv2d(self.total_rank, planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.basis_conv2 = nn.Conv2d(planes, rank, kernel_size=3, stride=stride, padding=1, bias=False)
        self.basis_bn2 = nn.BatchNorm2d(self.total_rank)
        self.coeff_conv2 = nn.Conv2d(self.total_rank, planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        #forward pass for basis trainig
        if (self.mode == 'train_basis'):
            #self.shared_basis.requires_grad = True
            #self.basis_conv1.requires_grad = True
            #self.basis_conv2.requires_grad = True
            #self.coeff_conv1.requires_grad = False
            #self.coeff_conv2.requires_grad = False
            
            out = self.basis_bn1(torch.cat((self.basis_conv1(x), self.shared_basis(x)),dim=1))
            out = F.relu(self.bn1(self.coeff_conv1(out)))
            out = self.bn2( self.coeff_conv2( self.basis_bn2(torch.cat((self.basis_conv2(out), self.shared_basis(out)),dim=1) ) ))
            out += self.shortcut(x)
            out = F.relu(out)
            return out
        #forward pass for coeff trainig
        #Feature maps from lower-than-50% coeffs are removed
        else:
            #self.shared_basis.requires_grad = False
            #self.basis_conv1.requires_grad = False
            #self.basis_conv2.requires_grad = False
            #self.coeff_conv1.requires_grad = True
            #self.coeff_conv2.requires_grad = True
            
            #mask_conv1 = torch.argsort(abs(self.coeff_conv1.weight).sum(dim=0).view(-1))
            
            out = torch.cat((self.basis_conv1(x), self.shared_basis(x)),dim=1)
            #out = nn.functional.conv2d(out[:,mask_conv1<self.rank,:,:], self.coeff_conv1.weight[:,mask_conv1<self.rank,:,:])
            out = F.relu(self.bn1(self.coeff_conv1(out)))
            #out = F.relu(self.bn1(out))
            
            #mask_conv2 = torch.argsort(abs(self.coeff_conv2.weight).sum(dim=0).view(-1))
            
            out = torch.cat((self.basis_conv2(out), self.shared_basis(out)),dim=1)
            #out = nn.functional.conv2d(out[:,mask_conv1<self.rank,:,:], self.coeff_conv2.weight[:,mask_conv2<self.rank,:,:])
            out = self.bn2(self.coeff_conv2(out))
            #out = self.bn2(out)
            
            out += self.shortcut(x)
            out = F.relu(out)
            return out
            
    
#Bottleneck Block with prameter sharing
#in_planes: Integer, number of input channels
#planes: Integer, number of output channels
#rank: Integer, rank of base(=total number of shared templates)
#shared_basis: tensor, tensor for shared base
#stride: Integer, stride
    
class Bottleneck_Basis(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, rank, shared_basis, stride=1):
        super(Bottleneck_Basis, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

#Basic Block without prameter sharing
#in_planes: Integer, number of input channels
#planes: Integer, number of output channels
#stride: Integer, stride
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

#Bottleneck Block without prameter sharing
#in_planes: Integer, number of input channels
#planes: Integer, number of output channels
#stride: Integer, stride
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.mode = 1
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet_Unique(nn.Module):
    def __init__(self, block, block_without_basis, num_blocks, num_classes, rank):
        super(ResNet_Unique, self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, block_without_basis, 64, num_blocks[0], rank, stride=1)
        self.layer2 = self._make_layer(block, block_without_basis, 128, num_blocks[1], rank*2, stride=2)
        self.layer3 = self._make_layer(block, block_without_basis, 256, num_blocks[2], rank*4, stride=2)
        self.layer4 = self._make_layer(block, block_without_basis, 512, num_blocks[3], rank*8, stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, block_without_basis, planes, num_blocks, rank, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        
        layers.append(block_without_basis(self.in_planes, planes, stride))
        self.in_planes = planes * block.expansion
            
        for stride in strides[1:]:
            layers.append(block(self.in_planes, planes, rank, stride))
            self.in_planes = planes * block.expansion
            
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
                           
#Resnet with prameter sharing
#block: Class, Residual block with parameter sharing
#blcok_without_basis: Class, Residual block without parameter sharing
#num_blocks: 4-elements list, number of blocks per group
#num_classes: Integer, total number of dataset's classes
#rank: Integer, rank of base(=total number of shared templates)

class ResNet_Basis(nn.Module):
    def __init__(self, block, block_without_basis, num_blocks, num_classes, rank, unique_rank):
        super(ResNet_Basis, self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.shared_basis_1 = nn.Conv2d(64, rank, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, block_without_basis, 64, num_blocks[0], unique_rank, self.shared_basis_1, stride=1)
        
        self.shared_basis_2 = nn.Conv2d(128, rank*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2 = self._make_layer(block, block_without_basis, 128, num_blocks[1], unique_rank*2, self.shared_basis_2, stride=2)
        
        self.shared_basis_3 = nn.Conv2d(256, rank*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3 = self._make_layer(block, block_without_basis, 256, num_blocks[2], unique_rank*4, self.shared_basis_3, stride=2)
        
        self.shared_basis_4 = nn.Conv2d(512, rank*8, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4 = self._make_layer(block, block_without_basis, 512, num_blocks[3], unique_rank*8, self.shared_basis_4, stride=2)
        
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, block_without_basis, planes, num_blocks, rank, basis, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        
        layers.append(block_without_basis(self.in_planes, planes, stride))
        self.in_planes = planes * block.expansion
            
        for stride in strides[1:]:
            layers.append(block(self.in_planes, planes, rank, basis, stride))
            self.in_planes = planes * block.expansion
            
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

#Parameter shared ResNet models
                           
def ResNet34_Unique(c, r):
    return ResNet_Unique(BasicBlock_Unique, BasicBlock, [3,4,6,3],c,r)
    
def ResNet18_Basis(c, r, u_r):
    return ResNet_Basis(BasicBlock_Basis, BasicBlock, [2,2,2,2],c,r, u_r)

def ResNet34_Basis(c, r, u_r):
    return ResNet_Basis(BasicBlock_Basis, BasicBlock, [3,4,6,3],c,r, u_r)

def ResNet50_Basis(c, r, u_r):
    return ResNet_Basis(Bottleneck_Basis, BasicBlock, [3,4,6,3],c,r, u_r)

def ResNet101_Basis(c, r, u_r):
    return ResNet_Basis(Bottleneck_Basis, BasicBlock, [3,4,23,3],c,r, u_r)

def ResNet152_Basis(c, r, u_r):
    return ResNet_Basis(Bottleneck_Basis, BasicBlock, [3,8,36,3],c,r, u_r)
