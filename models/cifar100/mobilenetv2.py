'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Blocks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class BlockShared(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride, shared_basis_1, shared_basis_2):
        super(BlockShared, self).__init__()
        self.stride = stride
        self.shared_basis1 = shared_basis_1
        self.shared_basis2 = shared_basis_2
        planes = expansion * in_planes

        self.unique_rank = planes - self.shared_basis2.weight.shape[0]
        
        #self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.basis_conv1 = nn.Conv2d(in_planes, self.unique_rank, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.basis_conv2 = nn.Conv2d(self.unique_rank, self.unique_rank, kernel_size=3, stride=stride, padding=1, groups=self.unique_rank, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        #print("x size #2:", x.shape)
        out = torch.cat((self.basis_conv1(x), self.shared_basis1(x)), dim=1)
        out = F.relu(self.bn1(out))

        out= torch.cat((self.basis_conv2(out[:,0:self.unique_rank,:]), self.shared_basis2(out[:,self.unique_rank:,:])), dim=1)
        out = F.relu(self.bn2(out))
 
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class MobileNetV2_Shared(nn.Module):

    def __init__(self, class_num=100):
        super(MobileNetV2_Shared, self).__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        self.stage1 = Block(32, 16, 1, 1)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)
        
        self.shared_basis_1_1=  nn.Conv2d(32, 31, kernel_size=1, stride=1, padding=0, bias=False)
        self.shared_basis_1_2 = nn.Conv2d(31, 31, kernel_size=3, stride=1, padding=1, groups=31, bias=False)        
        self.stage3 = self._make_shared_stage(3, 24, 32, 2, 6, self.shared_basis_1_1, self.shared_basis_1_2)

        self.shared_basis_2_1=  nn.Conv2d(64, 62, kernel_size=1, stride=1, padding=0, bias=False)
        self.shared_basis_2_2 = nn.Conv2d(62, 62, kernel_size=3, stride=1, padding=1, groups=62, bias=False)
        self.stage4 = self._make_shared_stage(4, 32, 64, 2, 6, self.shared_basis_2_1, self.shared_basis_2_2)

        self.shared_basis_3_1=  nn.Conv2d(96, 93, kernel_size=1, stride=1, padding=0, bias=False)
        self.shared_basis_3_2 = nn.Conv2d(93, 93, kernel_size=3, stride=1, padding=1, groups=93, bias=False)
        self.stage5 = self._make_shared_stage(3, 64, 96, 1, 6, self.shared_basis_3_1, self.shared_basis_3_2)

        self.shared_basis_4_1=  nn.Conv2d(160, 156, kernel_size=1, stride=1, padding=0, bias=False)
        self.shared_basis_4_2 = nn.Conv2d(156, 156, kernel_size=3, stride=1, padding=1, groups=156, bias=False)
        self.stage6 = self._make_shared_stage(3, 96, 160, 1, 6, self.shared_basis_4_1, self.shared_basis_4_2)

        self.stage7 = Block(160, 320, 6, 1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        self.conv2 = nn.Conv2d(1280, class_num, 1)

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        #print("x size #1:", x.shape)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        return x

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):

        layers = []
        layers.append(Block(in_channels, out_channels, t, stride))

        while repeat - 1:
            layers.append(Block(out_channels, out_channels, t, 1))
            repeat -= 1

        return nn.Sequential(*layers)

    def _make_shared_stage(self, repeat, in_channels, out_channels, stride, t, shared_basis1, shared_basis_2):

        layers = []
        layers.append(Block(in_channels, out_channels, t, stride))

        while repeat - 1:
            layers.append(BlockShared(out_channels, out_channels, t, 1, shared_basis1, shared_basis_2))
            repeat -= 1

        return nn.Sequential(*layers)

class MobileNetV2(nn.Module):

    def __init__(self, class_num=100):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        self.stage1 = Block(32, 16, 1, 1)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6)
        self.stage6 = self._make_stage(3, 96, 160, 1, 6)
        self.stage7 = Block(160, 320, 6, 1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        self.conv2 = nn.Conv2d(1280, class_num, 1)

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        return x

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):

        layers = []
        layers.append(Block(in_channels, out_channels, t, stride))

        while repeat - 1:
            layers.append(Block(out_channels, out_channels, t, 1))
            repeat -= 1

        return nn.Sequential(*layers)

def test():
    net = MobileNetV2_Shared(class_num=100)
    #print(net)
    x = torch.randn(256,3,32,32)
    y = net(x)
    print(y.size())
    #print(net)

test()

# class MobileNetV2(nn.Module):
#     # (expansion, out_planes, num_blocks, stride)
#     cfg = [(1,  16, 1, 1),
#            (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
#            (6,  32, 3, 2),
#            (6,  64, 4, 2),
#            (6,  96, 3, 1),
#            (6, 160, 3, 2),
#            (6, 320, 1, 1)]

#     def __init__(self, num_classes=100):
#         super(MobileNetV2, self).__init__()

#         #num_layers = sum([group[2] for group in self.cfg])
#         #self.basic_layers=[i for i in range(num_layers)]
#         self.basic_layers=[]
#         self.skip_layers=[]
#         self.skip_distance=[]

#         # NOTE: change conv1 stride 2 -> 1 for CIFAR10
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.layers = self._make_layers(in_planes=32)
#         self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn2 = nn.BatchNorm2d(1280)
#         self.linear = nn.Linear(1280, num_classes)
#         self.linear_skip = nn.Linear(1280, num_classes)

#     def _make_layers(self, in_planes):
#         layers = []
#         for expansion, out_planes, num_blocks, stride in self.cfg:
#             strides = [stride] + [1]*(num_blocks-1)
#             for stride in strides:
#                 layers.append(Block(in_planes, out_planes, expansion, stride))
#                 in_planes = out_planes
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layers(out)
#         out = F.relu(self.bn2(self.conv2(out)))
#         # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out


