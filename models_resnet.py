'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

g_rank= 8
g_d = 4

def decomposed_sequential_shared(layer, rank, d1, d2, d3):
    #decomp_1 = torch.nn.Conv2d(in_channels=layer.weight.shape[1], out_channels=rank, kernel_size=1, stride=1, padding=0, dilation=layer.dilation, bias=False)
    decomp_1 = d1
    bn1 = nn.BatchNorm2d(rank)
    
    #decomp_2 = torch.nn.Conv2d(in_channels=rank, out_channels=rank, kernel_size=(layer.weight.shape[2], 1), stride=1, padding=(layer.padding[0], 0), dilation=layer.dilation, groups=rank, bias=False)
    decomp_2 = d2
    bn2 = nn.BatchNorm2d(rank)
    
    #decomp_3 = torch.nn.Conv2d(in_channels=rank, out_channels=rank, kernel_size=(1, layer.weight.shape[3]), stride=1, padding=(0, layer.padding[0]), dilation=layer.dilation, groups=rank, bias=False)
    decomp_3 = d3
    bn3 = nn.BatchNorm2d(rank)
    
    decomp_4 = torch.nn.Conv2d(in_channels=rank, out_channels=layer.weight.shape[0]//g_d, kernel_size=1, stride=layer.stride,padding=0, dilation=layer.dilation, bias=False)
    bn4 = nn.BatchNorm2d(layer.weight.shape[0]//g_d)
    
    new_layers = [decomp_1, bn1, decomp_2, bn2, decomp_3, bn3, decomp_4, bn4]

    return nn.Sequential(*new_layers)

def decomposed_sequential(layer, rank):
    decomp_1 = torch.nn.Conv2d(in_channels=layer.weight.shape[1], out_channels=rank, kernel_size=1, stride=1, padding=0, dilation=layer.dilation, bias=False)
    bn1 = nn.BatchNorm2d(rank)
    decomp_2 = torch.nn.Conv2d(in_channels=rank, out_channels=rank, kernel_size=(layer.weight.shape[2], 1), stride=1, padding=(layer.padding[0], 0), dilation=layer.dilation, groups=rank, bias=False)
    bn2 = nn.BatchNorm2d(rank)
    decomp_3 = torch.nn.Conv2d(in_channels=rank, out_channels=rank, kernel_size=(1, layer.weight.shape[3]), stride=1, padding=(0, layer.padding[0]), dilation=layer.dilation, groups=rank, bias=False)
    bn3 = nn.BatchNorm2d(rank)
    decomp_4 = torch.nn.Conv2d(in_channels=rank, out_channels=layer.weight.shape[0]//g_d, kernel_size=1, stride=layer.stride,padding=0, dilation=layer.dilation, bias=False)
    bn4 = nn.BatchNorm2d(layer.weight.shape[0]//g_d)
    
    new_layers = [decomp_1, bn1, decomp_2, bn2, decomp_3, bn3, decomp_4, bn4]

    return nn.Sequential(*new_layers)

class BasicBlock_Decomposed_Shared(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, d1,d2,d3, stride=1):
        super(BasicBlock_Decomposed_Shared, self).__init__()
        if stride == 1:
            self.conv11 = decomposed_sequential_shared(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),g_rank, d1, d2, d3)
            self.conv12 = decomposed_sequential_shared(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),g_rank, d1, d2, d3)
            self.conv13 = decomposed_sequential_shared(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),g_rank, d1, d2, d3)
            self.conv14 = decomposed_sequential_shared(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),g_rank, d1, d2, d3)
        else:
            self.conv11 = decomposed_sequential(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),g_rank)
            self.conv12 = decomposed_sequential(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),g_rank)
            self.conv13 = decomposed_sequential(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),g_rank)
            self.conv14 = decomposed_sequential(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),g_rank)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.conv21 = decomposed_sequential_shared(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),g_rank, d1, d2, d3)
        self.conv22 = decomposed_sequential_shared(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),g_rank, d1, d2, d3)
        self.conv23 = decomposed_sequential_shared(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),g_rank, d1, d2, d3)
        self.conv24 = decomposed_sequential_shared(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),g_rank, d1, d2, d3)
        #self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu( torch.cat((self.conv11(x), self.conv12(x), self.conv13(x), self.conv14(x)), dim=1) )
        out = torch.cat((self.conv21(out), self.conv22(out), self.conv23(out), self.conv24(out)), dim=1)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock_Decomposed(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_Decomposed, self).__init__()
        self.conv11 = decomposed_sequential(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),g_rank)
        self.conv12 = decomposed_sequential(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),g_rank)
        self.conv13 = decomposed_sequential(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),g_rank)
        self.conv14 = decomposed_sequential(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),g_rank)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.conv21 = decomposed_sequential(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),g_rank)
        self.conv22 = decomposed_sequential(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),g_rank)
        self.conv23 = decomposed_sequential(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),g_rank)
        self.conv24 = decomposed_sequential(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),g_rank)
        #self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu( torch.cat((self.conv11(x), self.conv12(x), self.conv13(x), self.conv14(x)), dim=1) )
        out = torch.cat((self.conv21(out), self.conv22(out), self.conv23(out), self.conv24(out)), dim=1)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
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

class ResNet_shared(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet_shared, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.d1_1 = torch.nn.Conv2d(in_channels=64, out_channels=g_rank, kernel_size=1, stride=1, padding=0, bias=False)
        self.d2_1 = torch.nn.Conv2d(in_channels=g_rank, out_channels=g_rank, kernel_size=(3, 1), stride=1, padding=(1, 0), groups=g_rank, bias=False)
        self.d3_1 = torch.nn.Conv2d(in_channels=g_rank, out_channels=g_rank, kernel_size=(1, 3), stride=1, padding=(0, 1), groups=g_rank, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], self.d1_1,self.d2_1,self.d3_1, stride=1)
        
        self.d1_2 = torch.nn.Conv2d(in_channels=128, out_channels=g_rank, kernel_size=1, stride=1, padding=0, bias=False)
        self.d2_2 = torch.nn.Conv2d(in_channels=g_rank, out_channels=g_rank, kernel_size=(3, 1), stride=1, padding=(1, 0), groups=g_rank, bias=False)
        self.d3_2 = torch.nn.Conv2d(in_channels=g_rank, out_channels=g_rank, kernel_size=(1, 3), stride=1, padding=(0, 1), groups=g_rank, bias=False)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], self.d1_2,self.d2_2,self.d3_2, stride=2)
        
        self.d1_3 = torch.nn.Conv2d(in_channels=256, out_channels=g_rank, kernel_size=1, stride=1, padding=0, bias=False)
        self.d2_3 = torch.nn.Conv2d(in_channels=g_rank, out_channels=g_rank, kernel_size=(3, 1), stride=1, padding=(1, 0), groups=g_rank, bias=False)
        self.d3_3 = torch.nn.Conv2d(in_channels=g_rank, out_channels=g_rank, kernel_size=(1, 3), stride=1, padding=(0, 1), groups=g_rank, bias=False)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], self.d1_3,self.d2_3,self.d3_3, stride=2)
        
        self.d1_4 = torch.nn.Conv2d(in_channels=512, out_channels=g_rank, kernel_size=1, stride=1, padding=0, bias=False)
        self.d2_4 = torch.nn.Conv2d(in_channels=g_rank, out_channels=g_rank, kernel_size=(3, 1), stride=1, padding=(1, 0), groups=g_rank, bias=False)
        self.d3_4 = torch.nn.Conv2d(in_channels=g_rank, out_channels=g_rank, kernel_size=(1, 3), stride=1, padding=(0, 1), groups=g_rank, bias=False)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], self.d1_4,self.d2_4,self.d3_4, stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, d1,d2,d3, stride=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes,d1,d2,d3, stride))
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

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
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


def ResNet18(c):
    return ResNet(BasicBlock, [2,2,2,2],c)

def ResNet34_shared(c):
    return ResNet_shared(BasicBlock_Decomposed_Shared, [3,4,6,3],c)

def ResNet34_dc(c):
    return ResNet(BasicBlock_Decomposed, [3,4,6,3],c)

def ResNet34(c):
    return ResNet(BasicBlock, [3,4,6,3],c)

def ResNet50(c):
    return ResNet(Bottleneck, [3,4,6,3],c)

def ResNet101(c):
    return ResNet(Bottleneck, [3,4,23,3],c)

def ResNet152(c):
    return ResNet(Bottleneck, [3,8,36,3],c)


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()