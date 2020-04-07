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

from models.cifar import resnet, resnet_basis
import utils

#Possible arguments
parser = argparse.ArgumentParser(description='TODO')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--lambda2', default=0.5, type=float, help='lambda2 (for basis loss)')
parser.add_argument('--shared_rank', default=16, type=int, help='number of shared base)')
parser.add_argument('--dataset', default="CIFAR100", help='CIFAR10, CIFAR100')
parser.add_argument('--batch_size', default=256, type=int, help='batch_size')
parser.add_argument('--model', default="ResNet34", help='ResNet152, ResNet101, ResNet50, ResNet34, ResNet18, ResNet34_Basis, ResNet18_Basis')
parser.add_argument('--visible_device', default="0", help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--unique_rank', default=16, type=int, help='number of unique base')
args = parser.parse_args()

lr = args.lr
momentum = args.momentum
weight_decay = args.weight_decay
lambda2 = args.lambda2
rank = args.rank
unique_rank = args.unique_rank

dic_dataset = {'CIFAR100':100, 'CIFAR10':10}
dic_model = {'ResNet152':resnet.ResNet152,'ResNet101':resnet.ResNet101,'ResNet50':resnet.ResNet50,'ResNet34':resnet.ResNet34,'ResNet18':resnet.ResNet18,'ResNet34_Basis':resnet_basis.ResNet34_Basis,'ResNet18_Basis':resnet_basis.ResNet18_Basis}#, 'ResNet34_Unique':resnet_basis.ResNet34_Unique}

if args.dataset not in dic_dataset:
    print("The dataset is currently not supported")
    sys.exit()

if args.model not in dic_model:
    print("The model is currently not supported")
    sys.exit()

trainloader = utils.get_traindata(args.dataset,"./data",batch_size=args.batch_size,download=True)
testloader = utils.get_testdata(args.dataset,"./data",batch_size=args.batch_size)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]=args.visible_device
device='cuda'
#args.visible_device sets which cuda devices to be used"

if 'Basis' or 'Unique' in args.model:
    net = dic_model[args.model](dic_dataset[args.dataset], rank, unique_rank)
else:
    net = dic_model[args.model](dic_dataset[args.dataset])
    
net = net.to(device)

#CrossEntropyLoss for accuracy loss criterion
criterion = nn.CrossEntropyLoss()

#Unused - reserved for different LR schedulers
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#Training for standard models
def train(epoch):
    print('\nCuda ' + args.visible_device + ' Epoch: %d' % epoch)
    net.train()
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
    
        optimizer.zero_grad()
        outputs = net(inputs)
                        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
#Training for models with unique base only    
def train_unique(epoch):
    print("placeholder")
    
#Training for parameter shared models
def train_basis(epoch):
    print('\nCuda ' + args.visible_device + ' Basis Epoch: %d' % epoch)
    net.train()
    
    cos_simil= nn.CosineSimilarity(dim=-1)
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
    
        optimizer.zero_grad()
        outputs = net(inputs)
        
        sum_simil=0
        sum_cnt=0
        
        #absolute sum of cos similarity between every shared base
        #CosineSimilarity calculates cosine similarity of tensors along dim=1
        
        #group 1
        tmp_tuple_1 = (net.shared_basis_1.weight,)
        tmp_tuple_2 = (net.shared_basis_1.weight,)
        for i in range(1,len(net.layer1)):
            tmp_tuple_1 = tmp_tuple_1 + (net.layer1[i].basis_conv1.weight,)
            tmp_tuple_2 = tmp_tuple_2 + (net.layer1[i].basis_conv2.weight,)
        tmp_all_1 = torch.cat(tmp_tuple_1)
        tmp_all_2 = torch.cat(tmp_tuple_2)
        #tmp_all_1 and tmp_all2 contains every basis of residual blocks in group 1
        
        #for every basis in tmp_all_1 and tmp_all_2, calculates similarities to every other basis in the tmp_all tensor
        #tmp_all_1[i+1:]=
        #skips calculating itself (which the result is always 1.0)
        #skips already calculated similarity
        for i in range(tmp_all_1.shape[0]-1):
            tmp_1 = abs(cos_simil(
                tmp_all_1[i+1:].view(tmp_all_1.shape[0]-1-i,-1),
                tmp_all_1[i].view(-1)
            ))
            tmp_2 = abs(cos_simil(
                tmp_all_2[i+1:].view(tmp_all_2.shape[0]-1-i,-1),
                tmp_all_2[i].view(-1)
            ))
            #sum_simil contains every absolute sum of cos similarity
            #sum_cnt contains total number of base
            sum_simil=sum_simil + torch.sum(tmp_1)
            sum_cnt=sum_cnt + tmp_1.shape[0]
            sum_simil=sum_simil + torch.sum(tmp_2)
            sum_cnt=sum_cnt + tmp_2.shape[0]
            
        #group 2
        tmp_tuple_1 = (net.shared_basis_2.weight,)
        tmp_tuple_2 = (net.shared_basis_2.weight,)
        for i in range(1,len(net.layer2)):
            tmp_tuple_1 = tmp_tuple_1 + (net.layer2[i].basis_conv1.weight,)
            tmp_tuple_2 = tmp_tuple_2 + (net.layer2[i].basis_conv2.weight,)
        tmp_all_1 = torch.cat(tmp_tuple_1)
        tmp_all_2 = torch.cat(tmp_tuple_2)
        #tmp_all_1 and tmp_all2 contains every basis of residual blocks in group 2
        
        #for every basis in tmp_all_1 and tmp_all_2, calculates similarities to every other basis in the tmp_all tensor
        #tmp_all_1[i+1:]=
        #skips calculating itself (which the result is always 1.0)
        #skips already calculated similarity

        for i in range(tmp_all_1.shape[0]-1):
            tmp_1 = abs(cos_simil(
                tmp_all_1[i+1:].view(tmp_all_1.shape[0]-1-i,-1),
                tmp_all_1[i].view(-1)
            ))
            tmp_2 = abs(cos_simil(
                tmp_all_2[i+1:].view(tmp_all_2.shape[0]-1-i,-1),
                tmp_all_2[i].view(-1)
            ))
            #sum_simil contains every absolute sum of cos similarity
            #sum_cnt contains total number of base
            sum_simil=sum_simil + torch.sum(tmp_1)
            sum_cnt=sum_cnt + tmp_1.shape[0]
            sum_simil=sum_simil + torch.sum(tmp_2)
            sum_cnt=sum_cnt + tmp_2.shape[0]
            
        #group 3
        tmp_tuple_1 = (net.shared_basis_3.weight,)
        tmp_tuple_2 = (net.shared_basis_3.weight,)
        for i in range(1,len(net.layer3)):
            tmp_tuple_1 = tmp_tuple_1 + (net.layer3[i].basis_conv1.weight,)
            tmp_tuple_2 = tmp_tuple_2 + (net.layer3[i].basis_conv2.weight,)
        tmp_all_1 = torch.cat(tmp_tuple_1)
        tmp_all_2 = torch.cat(tmp_tuple_2)
        #tmp_all_1 and tmp_all2 contains every basis of residual blocks in group 3
        
        #for every basis in tmp_all_1 and tmp_all_2, calculates similarities to every other basis in the tmp_all tensor
        #tmp_all_1[i+1:]=
        #skips calculating itself (which the result is always 1.0)
        #skips already calculated similarity

        for i in range(tmp_all_1.shape[0]-1):
            tmp_1 = abs(cos_simil(
                tmp_all_1[i+1:].view(tmp_all_1.shape[0]-1-i,-1),
                tmp_all_1[i].view(-1)
            ))
            tmp_2 = abs(cos_simil(
                tmp_all_2[i+1:].view(tmp_all_2.shape[0]-1-i,-1),
                tmp_all_2[i].view(-1)
            ))
            #sum_simil contains every absolute sum of cos similarity
            #sum_cnt contains total number of base
            sum_simil=sum_simil + torch.sum(tmp_1)
            sum_cnt=sum_cnt + tmp_1.shape[0]
            sum_simil=sum_simil + torch.sum(tmp_2)
            sum_cnt=sum_cnt + tmp_2.shape[0]
            
        #group 4
        tmp_tuple_1 = (net.shared_basis_4.weight,)
        tmp_tuple_2 = (net.shared_basis_4.weight,)
        for i in range(1,len(net.layer4)):
            tmp_tuple_1 = tmp_tuple_1 + (net.layer4[i].basis_conv1.weight,)
            tmp_tuple_2 = tmp_tuple_2 + (net.layer4[i].basis_conv2.weight,)
        tmp_all_1 = torch.cat(tmp_tuple_1)
        tmp_all_2 = torch.cat(tmp_tuple_2)
        #tmp_all_1 and tmp_all2 contains every basis of residual blocks in group 4
        
        #for every basis in tmp_all_1 and tmp_all_2, calculates similarities to every other basis in the tmp_all tensor
        #tmp_all_1[i+1:]=
        #skips calculating itself (which the result is always 1.0)
        #skips already calculated similarity
        
        for i in range(tmp_all_1.shape[0]-1):
            tmp_1 = abs(cos_simil(
                tmp_all_1[i+1:].view(tmp_all_1.shape[0]-1-i,-1),
                tmp_all_1[i].view(-1)
            ))
            tmp_2 = abs(cos_simil(
                tmp_all_2[i+1:].view(tmp_all_2.shape[0]-1-i,-1),
                tmp_all_2[i].view(-1)
            ))
            #sum_simil contains every absolute sum of cos similarity
            #sum_cnt contains total number of base
            sum_simil=sum_simil + torch.sum(tmp_1)
            sum_cnt=sum_cnt + tmp_1.shape[0]
            sum_simil=sum_simil + torch.sum(tmp_2)
            sum_cnt=sum_cnt + tmp_2.shape[0]

        #average of sum_simil across every base
        sum_simil = sum_simil/sum_cnt

        #acc loss
        loss = criterion(outputs, targets)
        if (batch_idx == 0):
            print("accuracy_loss: %.10f" % loss)
        #apply similarity loss, multiplied by lambda2
        loss = loss - lambda2*torch.log(1 - sum_simil)
        if (batch_idx == 0):
            print("simililarity_loss: %.10f" % torch.log(1 - sum_simil))
            print("similarity:%.3f" % sum_simil)
        loss.backward()
        optimizer.step()
    
#Test for models
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
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc_top1,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt' + args.visible_device + '.pth')
        best_acc = acc_top1
        best_acc_top5 = acc_top5
        print("Best_Acc_top1 = %.3f" % acc_top1)
        print("Best_Acc_top5 = %.3f" % acc_top5)
        
best_acc = 0
best_acc_top5 = 0

#For parameter shared models, training happens in two rotating stages: basis training - coeff training - basis trianig - coeff training - ...
if 'Basis' in args.model:
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    for i in range(150):
        train_basis(i+1)
        test(i+1)
    
    #============
    
    checkpoint = torch.load('./checkpoint/ckpt' + args.visible_device + '.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    optimizer = optim.SGD(net.parameters(), lr=lr*0.1, momentum=momentum, weight_decay=weight_decay)
    for i in range(75):
        train_basis(i+151)
        test(i+151)
    
    #============
    
    checkpoint = torch.load('./checkpoint/ckpt' + args.visible_device + '.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    optimizer = optim.SGD(net.parameters(), lr=lr*0.01, momentum=momentum, weight_decay=weight_decay)
    for i in range(75):
        train_basis(i+226)
        test(i+226)

    print("Best_Acc_top1 = %.3f" % best_acc)
    print("Best_Acc_top5 = %.3f" % best_acc_top5)
    
elif 'Unique' in args.model:
    print("Placeholder")
    
else:
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    for i in range(80):
        train(i+1)
        test(i+1)
    
    #============
    
    checkpoint = torch.load('./checkpoint/ckpt' + args.visible_device + '.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    optimizer = optim.SGD(net.parameters(), lr=lr*0.1, momentum=momentum, weight_decay=weight_decay)
    for i in range(40):
        train(i+81)
        test(i+81)
    
    #============
    
    checkpoint = torch.load('./checkpoint/ckpt' + args.visible_device + '.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    optimizer = optim.SGD(net.parameters(), lr=lr*0.01, momentum=momentum, weight_decay=weight_decay)
    for i in range(40):
        train(i+121)
        test(i+121)

    print("Best_Acc_top1 = %.3f" % best_acc)
    print("Best_Acc_top5 = %.3f" % best_acc_top5)