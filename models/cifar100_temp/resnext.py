import torch
import torch.nn as nn
import torch.nn.functional as F
    
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

class ResNext(nn.Module):
    def __init__(self, block, num_blocks, num_classes, groups, base_width):
        super(ResNext, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], groups, base_width, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], groups, base_width, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], groups, base_width, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], groups, base_width, stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, groups, base_width, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, groups, base_width, stride))
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


def ResNext50_32x4d(c):
    return ResNext(BottleNeck, [3,4,6,3],c,32,4)

def ResNext101_32x8d(c):
    return ResNext(BottleNeck, [3,4,23,3],c,32,8)
