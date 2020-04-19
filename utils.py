import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

#Calculates meanstd of dataset
#dataset: String, of the dataset
#root: String, path of dataset
def dataset_meanstd(dataset, root, train=True, download=False):
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    data = torchvision.datasets.__dict__[dataset](root=root, train=train, download=download, transform=transform)
    
    mean = data.data.mean(axis=(0,1,2))
    std = data.data.std(axis=(0,1,2))
    
    mean = mean / 255
    std = std / 255
    
    return mean, std

#Prepares training dataset
#dataset: String, name of dataset
#root: String, path of dataset
def get_traindata(dataset, root, train=True, download=False, shuffle=True, batch_size=128, num_workers=8, crop_size=32, padding=4):
        
    if dataset=='CIFAR10':
        mean, std = dataset_meanstd(dataset, root, train=train, download=download)
        transform = transforms.Compose([
            transforms.RandomCrop(crop_size, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])
        
        data = torchvision.datasets.__dict__[dataset](root=root, train=train, download=download, transform=transform)
        trainloader = torch.utils.data.DataLoader(data,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=True)
        
    elif dataset=='CIFAR100':
        mean, std = dataset_meanstd(dataset, root, train=train, download=download)
        transform = transforms.Compose([
            transforms.RandomCrop(crop_size, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])
        
        data = torchvision.datasets.__dict__[dataset](root=root, train=train, download=download, transform=transform)
        trainloader = torch.utils.data.DataLoader(data,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=True)
        
    elif dataset=='ILSVRC2012':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        
        data = datasets.ImageFolder(
        root,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
        
        trainloader = torch.utils.data.DataLoader(data,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=True)
        
    else:
        print('Not implemented')
        trainloader = None
    
    return trainloader

#Prepares test dataset
#dataset: String, name of dataset
#root: String, path of dataset
def get_testdata(dataset, root, train=False, download=False, shuffle=False, batch_size=128, num_workers=8):
    
    if dataset=='CIFAR10':
        mean, std = dataset_meanstd(dataset, root, train=train, download=download)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])
        data = torchvision.datasets.__dict__[dataset](root=root, train=train, download=download, transform=transform)
        testloader = torch.utils.data.DataLoader(data,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=True)
    
    elif dataset=='CIFAR100':
        mean, std = dataset_meanstd(dataset, root, train=train, download=download)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])
        data = torchvision.datasets.__dict__[dataset](root=root, train=train, download=download, transform=transform)
        testloader = torch.utils.data.DataLoader(data,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=True)
    
    elif dataset=='ILSVRC2012':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        
        testloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(root, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, pin_memory=True)
        
    else:
        print('Not implemented')
        testloader = None
        
    return testloader

#Placeholder
def get_valdata():
    
    return
