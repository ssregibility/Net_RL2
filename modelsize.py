
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import sys
import os
import time

import utils
import datetime

import numpy as np

#from models.cifar100 import mobilenetv2
#from models.ilsvrc import mobilenetv2
from models.ilsvrc import resnet

from ptflops import get_model_complexity_info

#model = mobilenetv2_exp2.MobileNetV2_Shared
#model = mobilenetv2.MobileNetV2_Shared
#model = mobilenetv2.MobileNetV2
#model = resnet.ResNet34_SingleShared
#model = mobilenetv2.MobileNetV2_Shared
model = mobilenetv2.MobileNetV2_SharedDouble
#model = mobilenetv2.MobileNetV2

#model = torchvision.models.mobilenet_v2
#model = resnet.ResNet50_Shared
#model = resnet.ResNet50

with torch.cuda.device(0):
  net = model()
  #net = model(32,1)
  net = net.to('cuda')
  #inputsize = (3,32,32)
  inputsize = (3,224,224)
  macs, params = get_model_complexity_info(net, inputsize, as_strings=True,
                                           print_per_layer_stat=True, verbose=False)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))

#   x = torch.randn(256,3,32,32, device='cuda')

#   y = net(x)
#   t_start = time.time()
#   for i in range(100):
#       y = net(x)
#   t_end = time.time()
#   print('time: {:.3f} seconds per inference'.format((t_end - t_start)/100.0))
