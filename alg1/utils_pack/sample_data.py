import torch
from torchvision import datasets, transforms
import torchvision

# import sys
# sys.path.append('../')
from utils_pack.dataloader import MyDataset

def sample_fashion_mnist_data(N, device, num_labels=10,
                    data_dir='datasets/fashion_mnist', train=False, shuffle=True, 
                    model=None, x=None, y=None):
    if x is None and y is None:
        data_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(data_dir, train=train, download=True,
                            transform= transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.0,), (1.0,))
                            ])),
                            batch_size=N, shuffle=shuffle)
        iterater = iter(data_loader)
        x,y = iterater.next()
    x,y = x.to(device), y.to(device)
    # num = N
    rand_label = torch.randint(num_labels-1,[N],dtype=torch.long, device=device) + 1 
    #range from 1 to num_labels-1
    target_label = torch.fmod(y+rand_label, num_labels)
    if not model is None:
        out = model(x)
        pred = out.argmax(dim=1)
        idx = (pred == y)
        # num = idx.sum()
        x = x[idx]
        y = y[idx]
        target_label = target_label[idx]
        print('remained fraction: %.4f' % idx.float().mean())

    return x,y,target_label   

def sample_mnist_data(N, device, num_labels=10,
                    data_dir='datasets/mnist', train=False, shuffle=True, 
                    model=None, x=None, y=None):
    if x is None and y is None:
        data_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_dir, train=train, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.0,), (1.0,))
                        ])),
            batch_size=N, shuffle=shuffle)
        iterater = iter(data_loader)
        x,y = iterater.next()
    x,y = x.to(device), y.to(device)
    # num = N
    rand_label = torch.randint(num_labels-1,[N],dtype=torch.long, device=device) + 1 
    #range from 1 to num_labels-1
    target_label = torch.fmod(y+rand_label, num_labels)
    if not model is None:
        out = model(x)
        pred = out.argmax(dim=1)
        idx = (pred == y)
        # num = idx.sum()
        x = x[idx]
        y = y[idx]
        target_label = target_label[idx]
        print('remained fraction: %.4f' % idx.float().mean())

    return x,y,target_label   