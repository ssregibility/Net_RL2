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

import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
import numpy as np

#Possible arguments
parser = argparse.ArgumentParser(description='Following arguments are used for the script')
parser.add_argument('--lr', default=0.1, type=float, help='Learning Rate')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--lambdaR', default=10, type=float, help='Lambda (Basis regularization)')
parser.add_argument('--shared_rank', default=16, type=int, help='Number of shared base)')
parser.add_argument('--unique_rank', default=1, type=int, help='Number of unique base')
parser.add_argument('--batch_size', default=256, type=int, help='Batch_size')
parser.add_argument('--visible_device', default="0", help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--pretrained', default=None, help='Path of a pretrained model file')
parser.add_argument('--starting_epoch', default=0, type=int, help='An epoch which model training starts')
parser.add_argument('--dataset_path', default="./data/", help='A path to dataset directory')
parser.add_argument('--model', default="ResNet34_SingleShared", help='ResNet18, ResNet34, ResNet34_SingleShared, ResNet34_NonShared, ResNet34_SharedOnly, DenseNet121, DenseNet121_SingleShared, ResNext50, ResNext50_SingleShared, MobileNetV2, MobileNetV2_Shared')
args = parser.parse_args()

from models.cifar100 import resnet, densenet, resnext, mobilenetv2
dic_model = {'ResNet18': resnet.ResNet18, \
        'ResNet34':resnet.ResNet34, \
        'ResNet34_SingleShared':resnet.ResNet34_SingleShared, \
        'ResNet34_NonShared':resnet.ResNet34_NonShared, \
        'ResNet34_SharedOnly':resnet.ResNet34_SharedOnly, \
        'DenseNet121':densenet.DenseNet121, \
        'DenseNet121_SingleShared':densenet.DenseNet121_SingleShared, \
        'ResNext50':resnext.ResNext50_32x4d, \
        'ResNext50_SingleShared':resnext.ResNext50_32x4d_SingleShared, \
        'MobileNetV2':mobilenetv2.MobileNetV2, \
        'MobileNetV2_Shared':mobilenetv2.MobileNetV2_Shared, \
        'MobileNetV2_SharedDouble':mobilenetv2.MobileNetV2_SharedDouble}        

if args.model not in dic_model:
    print("The model is currently not supported")
    sys.exit()

trainloader = utils.get_traindata('CIFAR100',args.dataset_path,batch_size=args.batch_size,download=True)
testloader = utils.get_testdata('CIFAR100',args.dataset_path,batch_size=args.batch_size)

#args.visible_device sets which cuda devices to be used
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]=args.visible_device
device='cuda'

if 'SingleShared' in args.model:
    net = dic_model[args.model](args.shared_rank, args.unique_rank)
elif 'SharedOnly' in args.model:
    net = dic_model[args.model](args.shared_rank)
elif 'NonShared' in args.model:
    net = dic_model[args.model](args.unique_rank)
else:
    net = dic_model[args.model]()
    
net = net.to(device)

#CrossEntropyLoss for accuracy loss criterion
criterion = nn.CrossEntropyLoss()

"""
Training original models.
"""
def train(epoch):

    print('\nCuda ' + args.visible_device + ' Epoch: %d' % epoch)
    net.train()
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
    
        optimizer.zero_grad()
        outputs = net(inputs)
        
        _, pred = outputs.topk(5, 1, largest=True, sorted=True)

        label_e = targets.view(targets.size(0), -1).expand_as(pred)
        correct = pred.eq(label_e).float()

        correct_top5 += correct[:, :5].sum()
        correct_top1 += correct[:, :1].sum()        
        total += targets.size(0)
                        
        loss = criterion(outputs, targets)
        if (batch_idx == 0):
            print("accuracy_loss: %.6f" % loss)
        loss.backward()
        optimizer.step()
        
    acc_top1 = 100.*correct_top1/total
    acc_top5 = 100.*correct_top5/total
    
    print("Training_Acc_Top1 = %.3f" % acc_top1)
    print("Training_Acc_Top5 = %.3f" % acc_top5)


"""
Training models that share a single-basis.
""" 
def train_basis(epoch):
    print('\nCuda ' + args.visible_device + ' Basis Epoch: %d' % epoch)
    net.train()
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
    
        optimizer.zero_grad()
        outputs = net(inputs)

        _, pred = outputs.topk(5, 1, largest=True, sorted=True)

        label_e = targets.view(targets.size(0), -1).expand_as(pred)
        correct = pred.eq(label_e).float()

        correct_top5 += correct[:, :5].sum()
        correct_top1 += correct[:, :1].sum()        
        total += targets.size(0)
        
        # get similarity of basis filters
        cnt_sim = 0 
        sim = 0
        for gid in range(1, 5):  # all models have 4 groups
            shared_basis = getattr(net,"shared_basis_"+str(gid))

            num_shared_basis = shared_basis.weight.shape[0]
            num_all_basis = num_shared_basis 

            all_basis =(shared_basis.weight,)

            B = torch.cat(all_basis).view(num_all_basis, -1)
            #print("B size:", B.shape)

            # compute orthogonalities btwn all baisis  
            D = torch.mm(B, torch.t(B)) 

            # make diagonal zeros
            D = (D - torch.eye(num_all_basis, num_all_basis, device=device))**2
            
            sim += torch.sum(D[0:num_shared_basis,0:num_shared_basis])
            cnt_sim += num_shared_basis**2

        #average similarity
        avg_sim = sim / cnt_sim

        #acc loss
        loss = criterion(outputs, targets)

        if (batch_idx == 0):
            print("accuracy_loss: %.6f" % loss)
            print("similarity loss: %.6f" % avg_sim)

        #apply similarity loss, multiplied by args.lambdaR
        loss = loss + avg_sim * args.lambdaR
        loss.backward()
        optimizer.step()
        
    acc_top1 = 100.*correct_top1/total
    acc_top5 = 100.*correct_top5/total
    
    print("Training_Acc_Top1 = %.3f" % acc_top1)
    print("Training_Acc_Top5 = %.3f" % acc_top5)


"""
Training models that have 2 bases in each residual block.
e.g., MobileNetV2 with CIFAR100 have 2 separate bases in each block
"""
def train_basis_double_separate(epoch):
    print('\nCuda ' + args.visible_device + ' Basis Epoch: %d' % epoch)
    net.train()
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    #plt.figure()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
    
        optimizer.zero_grad()
        outputs = net(inputs)

        _, pred = outputs.topk(5, 1, largest=True, sorted=True)

        label_e = targets.view(targets.size(0), -1).expand_as(pred)
        correct = pred.eq(label_e).float()

        correct_top5 += correct[:, :5].sum()
        correct_top1 += correct[:, :1].sum()        
        total += targets.size(0)
        
        # get similarity of basis filters
        cnt_sim = 0 
        sim = 0
        for gid in range(1, 5):  # all models have 4 groups
            shared_basis = getattr(net,"shared_basis_"+str(gid)+"_1")

            num_shared_basis = shared_basis.weight.shape[0]
            num_all_basis = num_shared_basis 

            all_basis =(shared_basis.weight,)

            B = torch.cat(all_basis).view(num_all_basis, -1)
            #print("B size:", B.shape)

            # compute orthogonalities btwn all baisis  
            D = torch.mm(B, torch.t(B)) 

            # make diagonal zeros
            D = (D - torch.eye(num_all_basis, num_all_basis, device=device))**2
            
            sim += torch.sum(D[0:num_shared_basis,0:num_shared_basis])
            cnt_sim += num_shared_basis**2

        #average similarity
        avg_sim_1 = sim / cnt_sim

        cnt_sim = 0 
        sim = 0
        for gid in range(1, 5):  # all models have 4 groups
            shared_basis = getattr(net,"shared_basis_"+str(gid)+"_2")

            num_shared_basis = shared_basis.weight.shape[0]
            num_all_basis = num_shared_basis 

            all_basis =(shared_basis.weight,)

            B = torch.cat(all_basis).view(num_all_basis, -1)
            #print("B size:", B.shape)

            # compute orthogonalities btwn all baisis  
            D = torch.mm(B, torch.t(B)) 

            # make diagonal zeros
            D = (D - torch.eye(num_all_basis, num_all_basis, device=device))**2
            
            sim += torch.sum(D[0:num_shared_basis,0:num_shared_basis])
            cnt_sim += num_shared_basis**2

        #average similarity
        avg_sim_2 = sim / cnt_sim

        #acc loss
        loss = criterion(outputs, targets)

        if (batch_idx == 0):
            print("accuracy_loss: %.6f" % loss)
            print("similarity loss: %.6f" % avg_sim_1)
            print("similarity loss: %.6f" % avg_sim_2)
        #apply similarity loss, multiplied by args.lambdaR
        loss = loss + (avg_sim_1+avg_sim_2) * args.lambdaR
        loss.backward()
        optimizer.step()
        
    acc_top1 = 100.*correct_top1/total
    acc_top5 = 100.*correct_top5/total
    
    print("Training_Acc_Top1 = %.3f" % acc_top1)
    print("Training_Acc_Top5 = %.3f" % acc_top5)


"""
Train original models & collect gradient data.

Note: Adust BNs in models/cifar100/resnet.py to see the effect of BNs.
Note: Use gradflow.ipynb notebook to visulize the data.
"""
def train_gradflow(epoch):
    print('\n[train]Cuda ' + args.visible_device + ' Epoch: %d' % epoch)
    net.train()
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
    
        optimizer.zero_grad()
        outputs = net(inputs)
        
        _, pred = outputs.topk(5, 1, largest=True, sorted=True)

        label_e = targets.view(targets.size(0), -1).expand_as(pred)
        correct = pred.eq(label_e).float()

        correct_top5 += correct[:, :5].sum()
        correct_top1 += correct[:, :1].sum()        
        total += targets.size(0)
                        
        loss = criterion(outputs, targets)
        if (batch_idx == 0):
            print("accuracy_loss: %.6f" % loss)
        loss.backward()

        if (batch_idx ==0 and epoch % 2 == 0):
            filename="experi/gradflow-noortho-nobn2"
            f_mean = open("{}-mean-{}.txt".format(filename, epoch),"w+")
            f_max = open("{}-max-{}.txt".format(filename, epoch),"w+")
        if (batch_idx % 10 ==0 and epoch % 2 == 0):
            ave_grads = ""
            max_grads = ""
            for n, p in net.named_parameters():
                if (p.requires_grad) and (("shared_basis" in n)):
                    ave_grads += "{}\t".format(p.grad.abs().mean())
                    max_grads += "{}\t".format(p.grad.abs().max())
            f_mean.write("{}\n".format(ave_grads))
            f_max.write("{}\n".format(max_grads))
        if (batch_idx == (len(trainloader)-1) and epoch % 2 == 0 ):
            f_mean.close()
            f_max.close()
        
        optimizer.step()
        
    acc_top1 = 100.*correct_top1/total
    acc_top5 = 100.*correct_top5/total
    
    print("Training_Acc_Top1 = %.3f" % acc_top1)
    print("Training_Acc_Top5 = %.3f" % acc_top5)

 
"""
Train single-basis models & collect gradient data

Note: Adust BNs in models/cifar100/resnet.py to see the effect of BNs.
Note: Use gradflow.ipynb notebook to visulize the data.
"""
def train_basis_gradflow(epoch):
    print('\n[train_basis]Cuda ' + args.visible_device + ' Basis Epoch: %d' % epoch)
    net.train()
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
    
        optimizer.zero_grad()
        outputs = net(inputs)

        _, pred = outputs.topk(5, 1, largest=True, sorted=True)

        label_e = targets.view(targets.size(0), -1).expand_as(pred)
        correct = pred.eq(label_e).float()

        correct_top5 += correct[:, :5].sum()
        correct_top1 += correct[:, :1].sum()        
        total += targets.size(0)
        
        # get similarity of basis filters
        cnt_sim = 0 
        sim = 0
        for gid in range(1, 5):  # all models have 4 groups
            shared_basis = getattr(net,"shared_basis_"+str(gid))

            num_shared_basis = shared_basis.weight.shape[0]
            num_all_basis = num_shared_basis 

            all_basis =(shared_basis.weight,)

            B = torch.cat(all_basis).view(num_all_basis, -1)
            #print("B size:", B.shape)

            # compute orthogonalities btwn all baisis  
            D = torch.mm(B, torch.t(B)) 

            # make diagonal zeros
            D = (D - torch.eye(num_all_basis, num_all_basis, device=device))**2
            
            sim += torch.sum(D[0:num_shared_basis,0:num_shared_basis])
            cnt_sim += num_shared_basis**2

        #average similarity
        avg_sim = sim / cnt_sim

        #acc loss
        loss = criterion(outputs, targets)

        if (batch_idx == 0):
            print("accuracy_loss: %.6f" % loss)
            print("similarity loss: %.6f" % avg_sim)

        #apply similarity loss, multiplied by args.lambdaR
        loss = loss + avg_sim * args.lambdaR
        loss.backward()

        if (batch_idx ==0 and epoch % 2 == 0):
            filename="experi/gradflow-ortho-nobn"
            f_mean = open("{}-mean-{}.txt".format(filename, epoch),"w+")
            f_max = open("{}-max-{}.txt".format(filename, epoch),"w+")
        if (batch_idx % 10 ==0 and epoch % 2 == 0):
            ave_grads = ""
            max_grads = ""
            for n, p in net.named_parameters():
                if (p.requires_grad) and (("shared_basis" in n)):
                    ave_grads += "{}\t".format(p.grad.abs().mean())
                    max_grads += "{}\t".format(p.grad.abs().max())
            f_mean.write("{}\n".format(ave_grads))
            f_max.write("{}\n".format(max_grads))
        if (batch_idx == (len(trainloader)-1) and epoch % 2 == 0 ):
            f_mean.close()
            f_max.close()

        optimizer.step()
        
    acc_top1 = 100.*correct_top1/total
    acc_top5 = 100.*correct_top5/total
    
    print("Training_Acc_Top1 = %.3f" % acc_top1)
    print("Training_Acc_Top5 = %.3f" % acc_top5)


'''
Test for models
'''
def test(epoch):
    global best_acc
    global best_acc_top5
    net.eval()
    test_loss = 0
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
    if acc_top1 > best_acc:
        #print('Saving..')
        state = {
            'net_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'acc': acc_top1,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + 'CIFAR100-' + args.model + "-S" + str(args.shared_rank) + "-U" + str(args.unique_rank) + "-L" + str(args.lambdaR) + "-" + args.visible_device + '.pth')
        best_acc = acc_top1
        best_acc_top5 = acc_top5
        print("Best_Acc_top1 = %.3f" % acc_top1)
        print("Best_Acc_top5 = %.3f" % acc_top5)
        

def adjust_learning_rate(optimizer, epoch, args_lr):
    lr = args_lr
    if epoch > 150:
        lr = lr * 0.1
    if epoch > 225:
        lr = lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
best_acc = 0
best_acc_top5 = 0

func_train = train
if 'SingleShared' in args.model or 'SharedOnly' in args.model or 'MobileNetV2_Shared' in args.model:
    func_train = train_basis
    #func_train = train
if 'MobileNetV2_SharedDouble' in args.model:
    func_train = train_basis_double_separate

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

if args.pretrained != None:
    checkpoint = torch.load(args.pretrained)
    net.load_state_dict(checkpoint['net_state_dict'])
    best_acc = checkpoint['acc']
                                
for i in range(args.starting_epoch, 300):
    start = timeit.default_timer()
    func_train(i+1)
    test(i+1)
    
    stop = timeit.default_timer()
    print('Time: ', stop - start)  

    #============
    
print("Best_Acc_top1 = %.3f" % best_acc)
print("Best_Acc_top5 = %.3f" % best_acc_top5)
