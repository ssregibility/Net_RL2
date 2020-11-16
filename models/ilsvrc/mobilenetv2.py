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

        self.use_res_connect = self.stride == 1 and in_planes == out_planes

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        #out = out + self.shortcut(x) if self.stride==1 else out
        if self.use_res_connect:
            out = x + out

        return out

class BlockShared(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride, shared_basis_1):
        super(BlockShared, self).__init__()
        self.stride = stride
        self.shared_basis1 = shared_basis_1
        planes = expansion * in_planes
 
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.use_res_connect = self.stride == 1 and in_planes == out_planes

    def forward(self, x):
        out = F.relu(self.bn1(self.shared_basis1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))

        if self.use_res_connect:
            out = x + out

        return out

class BlockSharedDouble(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride, shared_basis_1, shared_basis_2):
        super(BlockSharedDouble, self).__init__()
        self.stride = stride
        self.shared_basis1 = shared_basis_1
        self.shared_basis2 = shared_basis_2
        planes = expansion * in_planes
 
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        #self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.use_res_connect = self.stride == 1 and in_planes == out_planes

    def forward(self, x):
        out = F.relu(self.bn1(self.shared_basis1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.shared_basis2(out))

        if self.use_res_connect:
            out = x + out

        return out


class MobileNetV2_Shared(nn.Module):

    def __init__(self, class_num=1000):
        super(MobileNetV2_Shared, self).__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # 1x1->3x3, stride 1->2 for ilsvrc
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        self.stage1 = Block(32, 16, 1, 1)  # stride 2->1 for ilsvrc
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)
        
        self.shared_basis_1=  nn.Conv2d(32, 32*6, kernel_size=1, stride=1, padding=0, bias=False)
        self.stage3 = self._make_shared_stage(3, 24, 32, 2, 6, self.shared_basis_1)

        self.shared_basis_2=  nn.Conv2d(64, 64*6, kernel_size=1, stride=1, padding=0, bias=False)
        self.stage4 = self._make_shared_stage(4, 32, 64, 2, 6, self.shared_basis_2)

        self.shared_basis_3=  nn.Conv2d(96, 96*6, kernel_size=1, stride=1, padding=0, bias=False)
        self.stage5 = self._make_shared_stage(3, 64, 96, 1, 6, self.shared_basis_3)

        self.shared_basis_4=  nn.Conv2d(160, 160*6, kernel_size=1, stride=1, padding=0, bias=False)  
        self.stage6 = self._make_shared_stage(3, 96, 160, 2, 6, self.shared_basis_4)  # 1->2 for ilsvrc

        self.stage7 = Block(160, 320, 6, 1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),  
            nn.Linear(1280, class_num),
        )


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
        x = F.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)

        return x

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):

        layers = []
        layers.append(Block(in_channels, out_channels, t, stride))

        while repeat - 1:
            layers.append(Block(out_channels, out_channels, t, 1))
            repeat -= 1

        return nn.Sequential(*layers)

    def _make_shared_stage(self, repeat, in_channels, out_channels, stride, t, shared_basis1):

        layers = []
        layers.append(Block(in_channels, out_channels, t, stride))

        while repeat - 1:
            layers.append(BlockShared(out_channels, out_channels, t, 1, shared_basis1))
            repeat -= 1

        return nn.Sequential(*layers)

class MobileNetV2_SharedDouble(nn.Module):

    def __init__(self, class_num=1000):
        super(MobileNetV2_SharedDouble, self).__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # 1x1->3x3, stride 1->2 for ilsvrc
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        self.stage1 = Block(32, 16, 1, 1)  # stride 2->1 for ilsvrc
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)
        
        self.shared_basis_1_1=  nn.Conv2d(32, 32*6, kernel_size=1, stride=1, padding=0, bias=False)
        self.shared_basis_1_2 = nn.Conv2d(32*6, 32, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.stage3 = self._make_shared_stage(3, 24, 32, 2, 6, self.shared_basis_1_1, self.shared_basis_1_2)

        self.shared_basis_2_1=  nn.Conv2d(64, 64*6, kernel_size=1, stride=1, padding=0, bias=False)
        self.shared_basis_2_2 = nn.Conv2d(64*6, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.stage4 = self._make_shared_stage(4, 32, 64, 2, 6, self.shared_basis_2_1, self.shared_basis_2_2)

        self.shared_basis_3_1=  nn.Conv2d(96, 96*6, kernel_size=1, stride=1, padding=0, bias=False)
        self.shared_basis_3_2 = nn.Conv2d(96*6, 96, kernel_size=1, stride=1, padding=0, bias=False)
        self.stage5 = self._make_shared_stage(3, 64, 96, 1, 6, self.shared_basis_3_1, self.shared_basis_3_2)

        self.shared_basis_4_1=  nn.Conv2d(160, 160*6, kernel_size=1, stride=1, padding=0, bias=False)
        self.shared_basis_4_2 = nn.Conv2d(160*6, 160, kernel_size=1, stride=1, padding=0, bias=False)
        self.stage6 = self._make_shared_stage(3, 96, 160, 2, 6, self.shared_basis_4_1, self.shared_basis_4_2)  # 1->2 for ilsvrc

        self.stage7 = Block(160, 320, 6, 1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),  
            nn.Linear(1280, class_num),
        )


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
        x = F.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)

        return x

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):

        layers = []
        layers.append(Block(in_channels, out_channels, t, stride))

        while repeat - 1:
            layers.append(Block(out_channels, out_channels, t, 1))
            repeat -= 1

        return nn.Sequential(*layers)

    def _make_shared_stage(self, repeat, in_channels, out_channels, stride, t, shared_basis1, shared_basis2):

        layers = []
        layers.append(Block(in_channels, out_channels, t, stride))

        while repeat - 1:
            layers.append(BlockSharedDouble(out_channels, out_channels, t, 1, shared_basis1, shared_basis2))
            repeat -= 1

        return nn.Sequential(*layers)


class MobileNetV2(nn.Module):

    def __init__(self, class_num=1000):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # 1x1->3x3, stride 1->2 for ilsvrc
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        self.stage1 = Block(32, 16, 1, 1)  
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6)
        self.stage6 = self._make_stage(3, 96, 160, 2, 6)  # 1->2 for ilsvrc
        self.stage7 = Block(160, 320, 6, 1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),  
            nn.Linear(1280, class_num),
        )

        #self.conv2 = nn.Conv2d(1280, class_num, 1)

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
        x = F.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)

        return x

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):

        layers = []
        layers.append(Block(in_channels, out_channels, t, stride))


        while repeat - 1:
            layers.append(Block(out_channels, out_channels, t, 1))
            repeat -= 1

        return nn.Sequential(*layers)

def test():
    net = MobileNetV2_SharedDouble(class_num=1000)
    #net = MobileNetV2(class_num=1000)
    #print(net)
    #x = torch.randn(256,3,32,32)
    x = torch.randn(16,3,224,224)
    y = net(x)
    #print(y.size())
    print(net)

#test()

