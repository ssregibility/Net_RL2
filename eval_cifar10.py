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
parser = argparse.ArgumentParser(description='TODO')
parser.add_argument('--shared_rank', default=16, type=int, help='Number of shared base)')
parser.add_argument('--unique_rank', default=1, type=int, help='Number of unique base')
parser.add_argument('--batch_size', default=256, type=int, help='Batch_size')
parser.add_argument('--visible_device', default="0", help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--pretrained', default="./pretrained/CIFAR10_ResNet56_Double_S16U1_6.30err.pth", help='Path of a pretrained model file')
parser.add_argument('--dataset_path', default="./data/", help='A path to dataset directory')
parser.add_argument('--model', default="ResNet56_Basis", help='ResNet20, ResNet32, ResNet44, ResNet56, ResNet110, ResNet1202, ResNet56_Basis, ResNet32_Basis, ResNet56_Single, ResNet32_Single, ResNet56_Shared, ResNet32_Shared, ResNet56_Unique, ResNet32_Unique')
args = parser.parse_args()

from models.cifar10 import resnet
dic_model = {'ResNet20': resnet.ResNet20, 'ResNet32':resnet.ResNet32,'ResNet44':resnet.ResNet44,'ResNet56':resnet.ResNet56, 'ResNet110':resnet.ResNet110, 'ResNet1202':resnet.ResNet1202, 'ResNet56_Basis':resnet.ResNet56_Basis, 'ResNet32_Basis':resnet.ResNet32_Basis, 'ResNet56_Single':resnet.ResNet56_Single, 'ResNet32_Single':resnet.ResNet32_Single, 'ResNet56_Shared':resnet.ResNet56_Shared, 'ResNet32_Shared':resnet.ResNet32_Shared, 'ResNet56_Unique':resnet.ResNet56_Unique, 'ResNet32_Unique':resnet.ResNet32_Unique}
    
if args.model not in dic_model:
    print("The model is currently not supported")
    sys.exit()

testloader = utils.get_testdata('CIFAR10',args.dataset_path,batch_size=args.batch_size)

#args.visible_device sets which cuda devices to be used
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]=args.visible_device
device='cuda'

if 'Basis' in args.model or 'Single' in args.model:
    net = dic_model[args.model](args.shared_rank, args.unique_rank)
elif 'Shared' in args.model:
    net = dic_model[args.model](args.shared_rank)
elif 'Unique' in args.model:
    net = dic_model[args.model](args.unique_rank)
else:
    net = dic_model[args.model]()
    
net = net.to(device)

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