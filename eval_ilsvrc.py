from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import sys
import os
import argparse

import utils
import timeit

#Possible arguments
parser = argparse.ArgumentParser(description='Following arguments are used for the script')
parser.add_argument('--lr', default=0.1, type=float, help='Learning Rate')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay')
parser.add_argument('--lambdaR', default=10, type=float, help='Lambda (Basis regularization)')
parser.add_argument('--shared_rank', default=32, type=int, help='Number of shared base)')
parser.add_argument('--unique_rank', default=1, type=int, help='Number of unique base')
parser.add_argument('--batch_size', default=256, type=int, help='Batch_size')
parser.add_argument('--visible_device', default="0", help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--pretrained', default=None, help='Path of a pretrained model file')
parser.add_argument('--starting_epoch', default=0, type=int, help='An epoch which model training starts')
parser.add_argument('--dataset_path', default="/media/data/ILSVRC2012/", help='A path to dataset directory')
parser.add_argument('--model', default="ResNet34_DoubleShared", help='ResNet34, ResNet34_DoubleShared, ResNet34_SingleShared, ResNet50, ResNet50_Shared, MobileNetV2, MobileNetV2_Shared, MobileNetV2_SharedDouble')
args = parser.parse_args()

from models.ilsvrc import resnet, mobilenetv2

dic_model = {'ResNet18': resnet.ResNet18, \
    'ResNet34':resnet.ResNet34, \
    'ResNet34_DoubleShared':resnet.ResNet34_DoubleShared, \
    'ResNet34_SingleShared':resnet.ResNet34_SingleShared, \
    'ResNet50':resnet.ResNet50, \
    'ResNet50_Shared':resnet.ResNet50_Shared, \
    'ResNet50_SharedSingle':resnet.ResNet50_SharedSingle, \
    'ResNet101_Shared':resnet.ResNet101_Shared, \
    'ResNet101_SharedSingle':resnet.ResNet101_SharedSingle, \
    'MobileNetV2':mobilenetv2.MobileNetV2, \
    'MobileNetV2_Shared':mobilenetv2.MobileNetV2_Shared, \
    'MobileNetV2_SharedDouble':mobilenetv2.MobileNetV2_SharedDouble }
    
    
if args.model not in dic_model:
    print("The model is currently not supported")
    sys.exit()

testloader = utils.get_testdata('ILSVRC2012',args.dataset_path,batch_size=args.batch_size, num_workers=8)

#args.visible_device sets which cuda devices to be used
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]=args.visible_device
device='cuda'

if 'DoubleShared' in args.model or 'SingleShared' in args.model:
    net = dic_model[args.model](args.shared_rank, args.unique_rank)
else:
    net = dic_model[args.model]()
    
net = net.to(device)

# parallelize 
class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

# For ILSVRC, we use DataParallel in default
if torch.cuda.device_count() >= 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    net = MyDataParallel(net)


#Eval for models
def evaluation():
    net.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            
            _, pred = outputs.topk(5, 1, largest=True, sorted=True)

            label_e = targets.view(targets.size(0), -1).expand_as(pred)
            correct = pred.eq(label_e).float()

            correct_top5 += correct[:, :5].sum()
            correct_top1 += correct[:, :1].sum()
            
            total += targets.size(0)
            
    # Save checkpoint.
    acc_top1 = 100.*correct_top1/total
    acc_top5 = 100.*correct_top5/total

    print("Eval_Acc_top1 = %.3f" % acc_top1)
    print("Eval_Acc_top5 = %.3f" % acc_top5)
        
if args.pretrained != None:
    checkpoint = torch.load(args.pretrained)
    net.load_state_dict(checkpoint['net_state_dict'])
    
evaluation()
