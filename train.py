from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os

from models.cifar import resnet_basis
import utils


trainloader = utils.get_traindata('CIFAR100',"./data",batch_size=256,download=True)
testloader = utils.get_testdata('CIFAR100',"./data",batch_size=256)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device='cuda'

rank = 16

net = resnet_basis.ResNet34_Basis(100, rank)

net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

lmda1 = 0.1
lmda2 = 0.1

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_coeff(epoch):
    print('\nCuda0 Coeff Epoch: %d' % epoch)
    net.train()
    
    for i in net.layer1[1:]:
        i.mode = 'train_coeffs'
    for i in net.layer2[1:]:
        i.mode = 'train_coeffs'
    for i in net.layer3[1:]:
        i.mode = 'train_coeffs'
    for i in net.layer4[1:]:
        i.mode = 'train_coeffs'
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
    
        optimizer.zero_grad()
        outputs = net(inputs)
    
        sum_sharedcoeff = 0
        for i in net.layer1[1:]:
            sum_sharedcoeff = sum_sharedcoeff + torch.sum(abs(i.coeff_conv1.weight[:,rank:,:,:]))
            sum_sharedcoeff = sum_sharedcoeff + torch.sum(abs(i.coeff_conv2.weight[:,rank:,:,:]))
        for i in net.layer2[1:]:
            sum_sharedcoeff = sum_sharedcoeff + torch.sum(abs(i.coeff_conv1.weight[:,rank*2:,:,:]))
            sum_sharedcoeff = sum_sharedcoeff + torch.sum(abs(i.coeff_conv2.weight[:,rank*2:,:,:]))
        for i in net.layer3[1:]:
            sum_sharedcoeff = sum_sharedcoeff + torch.sum(abs(i.coeff_conv1.weight[:,rank*3:,:,:]))
            sum_sharedcoeff = sum_sharedcoeff + torch.sum(abs(i.coeff_conv2.weight[:,rank*3:,:,:]))
        for i in net.layer4[1:]:
            sum_sharedcoeff = sum_sharedcoeff + torch.sum(abs(i.coeff_conv1.weight[:,rank*4:,:,:]))
            sum_sharedcoeff = sum_sharedcoeff + torch.sum(abs(i.coeff_conv2.weight[:,rank*4:,:,:]))
                    
        loss = criterion(outputs, targets)
        loss = loss - lmda1*sum_sharedcoeff
        loss.backward()
        optimizer.step()
        
def train_basis(epoch):
    print('\nCuda0 Basis Epoch: %d' % epoch)
    net.train()
    
    for i in net.layer1[1:]:
        i.mode = 'train_basis'
    for i in net.layer2[1:]:
        i.mode = 'train_basis'
    for i in net.layer3[1:]:
        i.mode = 'train_basis'
    for i in net.layer4[1:]:
        i.mode = 'train_basis'
    
    cos_simil= nn.CosineSimilarity(dim=1)
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        print("!")
        inputs, targets = inputs.to(device), targets.to(device)
    
        optimizer.zero_grad()
        outputs = net(inputs)
        
        sum_simil=0
        sum_simil=sum_simil + torch.sum(cos_simil(net.shared_basis_1.weight[0].unsqueeze(dim=0),net.shared_basis_1.weight)[1:])
        sum_simil=sum_simil + torch.sum(cos_simil(net.shared_basis_2.weight[0].unsqueeze(dim=0),net.shared_basis_2.weight)[1:])
        sum_simil=sum_simil + torch.sum(cos_simil(net.shared_basis_3.weight[0].unsqueeze(dim=0),net.shared_basis_3.weight)[1:])
        sum_simil=sum_simil + torch.sum(cos_simil(net.shared_basis_4.weight[0].unsqueeze(dim=0),net.shared_basis_4.weight)[1:])
        
        loss = criterion(outputs, targets)
        loss = loss + lmda2*sum_simil
        loss.backward()
        optimizer.step()
    
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
        torch.save(state, './checkpoint/ckpt0_mult.pth')
        best_acc = acc_top1
        best_acc_top5 = acc_top5
        print("Best_Acc_top1 = %.3f" % acc_top1)
        print("Best_Acc_top5 = %.3f" % acc_top5)
        
best_acc = 0
best_acc_top5 = 0

optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
for i in range(150):
    train_basis(i+1)
    test(i+1)
    
checkpoint = torch.load('./checkpoint/ckpt0_mult.pth')
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']
    
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
for i in range(150):
    train_coeff(i+1)
    test(i+1)
    
#============
    
checkpoint = torch.load('./checkpoint/ckpt0_mult.pth')
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
for i in range(75):
    train_basis(i+151)
    test(i+151)
    
checkpoint = torch.load('./checkpoint/ckpt0_mult.pth')
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
for i in range(75):
    train_coeff(i+151)
    test(i+151)
    
#============
    
checkpoint = torch.load('./checkpoint/ckpt0_mult.pth')
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
for i in range(75):
    train_basis(i+226)
    test(i+226)

print("Best_Acc_top1 = %.3f" % best_acc)
print("Best_Acc_top5 = %.3f" % best_acc_top5)

checkpoint = torch.load('./checkpoint/ckpt0_mult.pth')
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
for i in range(75):
    train_coeff(i+226)
    test(i+226)

print("Best_Acc_top1 = %.3f" % best_acc)
print("Best_Acc_top5 = %.3f" % best_acc_top5)