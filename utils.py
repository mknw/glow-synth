import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.nn.functional import normalize

import numpy as np
import pandas as pd

def load_celeba(batch, image_size, test=False, shuffle=False):
    if not test:
        split = 'train'
        transform = transforms.Compose([
            transforms.CenterCrop(160),
            transforms.Resize(size=image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        shuffle=True
    else:
        split = 'test'
        transform = transforms.Compose([
        transforms.CenterCrop(160),
        transforms.Resize(size=image_size),
        transforms.ToTensor(),
    ])
    print(f'shuffle set to {shuffle} for split: {split}')
    target_type = ['attr', 'bbox', 'landmarks']
    dataset = datasets.CelebA(root='data', split=split, target_type=target_type[0], download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=8)
    return loader

def load_cifar_test(args):
    # Note: No normalization applied, since RealNVP expects inputs in (0, 1).
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    #torchvision.transforms.Normalize((0.1307,), (0.3081,)) # mean, std, inplace=False.
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return testloader

def load_mnist_test(args):
    transform_train = transforms.Compose([
        transforms.ToTensor()
        # transforms.ColorJitter(brightness=0.3)
    ])
    #torchvision.transforms.Normalize((0.1307,), (0.3081,)) # mean, std, inplace=False.
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    # trainset = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform_train)
    # trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testset = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return testloader

def load_network(model_dir, device, conf, checkpoint=True):
    if conf.arch == 'glow':
        from model import Glow
        net = Glow(3, conf.n_flows, conf.n_blocks, affine=conf.affine, conv_lu=not conf.no_lu)
        from train_like import calc_loss
        loss_fn = calc_loss
    elif conf.arch in ['densenet', 'resnet']:
        raise NotImplementedError

    net = net.to(device)
    if str(device).startswith('cuda'):
        net = torch.nn.DataParallel(net, conf.gpus)
        cudnn.benchmark = conf.benchmark

    # load checkpoint
    if checkpoint:
        checkpoint = torch.load(model_dir)
        try:
            net.load_state_dict(checkpoint['net'])
        except RuntimeError as re:
            print(re)
            raise ArchError('There is a problem importing the model, check parameters.')

    return net, loss_fn



class ArchError(Exception):
    def __init__(self, message=None):
        if not message:
            self.message = "State dictionary not matching your architecture. Check your params."
        else:
            self.message = message


if __name__ == '__main__':
    raise NotImplementedError
