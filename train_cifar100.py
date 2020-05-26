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
parser.add_argument('--lr', default=0.1, type=float, help='Learning Rate')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--lambdaR', default=10, type=float, help='Lambda (Basis regularization)')
parser.add_argument('--shared_rank', default=16, type=int, help='Number of shared base)')
parser.add_argument('--unique_rank', default=16, type=int, help='Number of unique base')
parser.add_argument('--batch_size', default=256, type=int, help='Batch_size')
parser.add_argument('--visible_device', default="0", help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--pretrained', default=None, help='Path of a pretrained model file')
parser.add_argument('--starting_epoch', default=0, type=int, help='An epoch which model training starts')
parser.add_argument('--dataset_path', default="./data/", help='Dataset path')
parser.add_argument('--model', default="ResNet56", help='ResNet20, ResNet32, ResNet44, ResNet56, ResNet110, ResNext1202')
args = parser.parse_args()

from models.cifar100 import resnet
dic_model = {'ResNet18': resnet.ResNet18, 'ResNet34':resnet.ResNet34, 'ResNet34_Basis':resnet.ResNet34_Basis}
    
if args.model not in dic_model:
    print("The model is currently not supported")
    sys.exit()

trainloader = utils.get_traindata('CIFAR100',args.dataset_path,batch_size=args.batch_size,download=True)
testloader = utils.get_testdata('CIFAR100',args.dataset_path,batch_size=args.batch_size)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]=args.visible_device
device='cuda'

#args.visible_device sets which cuda devices to be used
if 'Basis' in args.model:
    net = dic_model[args.model](args.shared_rank, args.unique_rank)
else:
    net = dic_model[args.model]()
    
net = net.to(device)

if args.pretrained != None:
    checkpoint = torch.load(args.pretrained)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
                    
#CrossEntropyLoss for accuracy loss criterion
criterion = nn.CrossEntropyLoss()

#Training for standard models
def train(epoch):
    if epoch < args.starting_epoch:
        return
    
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

# Training for parameter shraed models
# Use the property of orthogonal matrices;
# e.g.: AxA.T = I if A is orthogonal 
def train_basis(epoch, include_unique_basis=False):
    if epoch < args.starting_epoch:
        return
    
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
        for gid in range(1, 5):  # ResNet has 4 groups
            layer = getattr(net, "layer"+str(gid))
            shared_basis = getattr(net,"shared_basis_"+str(gid))

            num_shared_basis = shared_basis.weight.shape[0]
            num_all_basis = num_shared_basis 

            all_basis =(shared_basis.weight,)
            if (include_unique_basis == True):  
                num_unique_basis = layer[1].basis_conv1.weight.shape[0] 
                num_all_basis += (num_unique_basis * 2 * (len(layer) -1))
                for i in range(1, len(layer)):
                    all_basis += (layer[i].basis_conv1.weight, layer[i].basis_conv2.weight,)

            B = torch.cat(all_basis).view(num_all_basis, -1)
            #print("B size:", B.shape)

            # compute orthogonalities btwn all baisis  
            D = torch.mm(B, torch.t(B)) 

            # make diagonal zeros
            D = (D - torch.eye(num_all_basis, num_all_basis, device=device))**2
            
            #print("D size:", D.shape)
         
            if (include_unique_basis == True):  
                # orthogonalities btwn shared<->(shared/unique)
                sim += torch.sum(D[0:num_shared_basis,:])  
                cnt_sim += num_shared_basis*num_all_basis

                # orthogonalities btwn unique<->unique in the same layer
                for i in range(1, len(layer)):
                    for j in range(2):  # conv1 & conv2
                         idx_base = num_shared_basis   \
                          + (i-1) * (num_unique_basis) * 2 \
                          + num_unique_basis * j 
                         sim += torch.sum(\
                                 D[idx_base:idx_base + num_unique_basis, \
                                 idx_base:idx_base+num_unique_basis])
                         cnt_sim += num_unique_basis ** 2 
            else: # orthogonalities only btwn shared basis
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
        
#Test for models
def test(epoch):
    if epoch < args.starting_epoch:
        return
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
if 'Basis' in args.model:
    func_train = train_basis

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
"""
for i in range(300):
    start = timeit.default_timer()
    
    adjust_learning_rate(optimizer, i+1, args.lr)
    func_train(i+1)
    test(i+1)
    
    stop = timeit.default_timer()
    print('Time: ', stop - start)  

print("Best_Acc_top1 = %.3f" % best_acc)
print("Best_Acc_top5 = %.3f" % best_acc_top5)
"""

for i in range(150):
    start = timeit.default_timer()
    func_train(i+1)
    test(i+1)
    
    stop = timeit.default_timer()
    print('Time: ', stop - start)  

    #============
    
checkpoint = torch.load('./checkpoint/' + 'CIFAR100-' + args.model + "-S" + str(args.shared_rank) + "-U" + str(args.unique_rank) + "-L" + str(args.lambdaR) + "-" + args.visible_device + '.pth')
net.load_state_dict(checkpoint['net_state_dict'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

optimizer = optim.SGD(net.parameters(), lr=args.lr*0.1, momentum=args.momentum, weight_decay=args.weight_decay)
for i in range(75):
    start = timeit.default_timer()
    func_train(i+151)
    test(i+151)
    
    stop = timeit.default_timer()
    print('Time: ', stop - start)  
    
    #============
    
checkpoint = torch.load('./checkpoint/' + 'CIFAR100-' + args.model + "-S" + str(args.shared_rank) + "-U" + str(args.unique_rank) + "-L" + str(args.lambdaR) + "-" + args.visible_device + '.pth')
net.load_state_dict(checkpoint['net_state_dict'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

optimizer = optim.SGD(net.parameters(), lr=args.lr*0.01, momentum=args.momentum, weight_decay=args.weight_decay)
for i in range(75):
    start = timeit.default_timer()
    func_train(i+226)
    test(i+226)
    
    stop = timeit.default_timer()
    print('Time: ', stop - start)  

print("Best_Acc_top1 = %.3f" % best_acc)
print("Best_Acc_top5 = %.3f" % best_acc_top5)
