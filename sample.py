#!/var/scratch/mao540/miniconda3/envs/maip-venv/bin/python

import argparse
import os
import torch 
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
from tqdm import tqdm

from train_like import calc_z_shapes


def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() and len(args.gpu_ids) > 0 else "cpu")
    print("training on: %s" % device)
    start_epoch = 0
    args.model_dir = args.root_dir + '/epoch_150000'

    if args.net == 'glow':
        from model import Glow
        model = Glow(3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu)
        net = model.to(device)

    if str(device).startswith('cuda'):
        net = torch.nn.DataParallel(net, args.gpu_ids)
        cudnn.benchmark = args.benchmark

    # Load checkpoint.
    print('Resuming from checkpoint at ' + args.model_dir + '/model.pth.tar...')
    assert os.path.isdir(args.model_dir), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.model_dir + '/model.pth.tar')
    net.load_state_dict(checkpoint['net'])
    loss = checkpoint['test_loss']
    # we start epoch after the saved one (avoids overwrites).
    epoch = checkpoint['epoch']

    with torch.no_grad():
        sample_wrapper(args, net, device)


def sample_wrapper(args, net, device, noise=0.05, n_steps=2):
    # net.eval()
    save_path = args.root_dir + f'/n-{args.num_samples}_sample_t-{args.temp}'
    if args.img_size != 1:
        save_path += f'_hw-{args.img_size}'
    if noise:
        net = net_noise(net, device, noise)
        save_path += f'_eps-{noise}'

    # args.temp=temp
    x = sample(net, device, args, norm_img='std')
    print(f'init step: std: {x.std()}, mean: {x.mean()}')
    
    if n_steps:
        save_path += f'_s-{n_steps}'
        for s_i in range(n_steps):
            # x = (x - x.mean()) / x.std()
            x = resample(x, net, device, args, norm_img ='std')
            print(f'step{s_i}: std: {x.std()}, mean: {x.mean()}')

    # x = (x - x.min()) / (x.max() - x.min())
    save_path += '.png'
    torchvision.utils.save_image(x, save_path, normalize=True,
            nrow=int(args.num_samples ** 0.5), scale_each=True,
            pad_value=255) # , range=(-0.5, 0.5))
    print(f'saved to {save_path}')

def sample(net, device, args, norm_img=True, exp=False):
    """Sample from RealNVP model.

    Args:
        net (torch.nn.DataParallel): The RealNVP model wrapped in DataParallel.
        batch_size (int): Number of samples to generate.
        device (torch.device): Device to use.
    """
    if not exp:
        z_sample = []
        z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
        for z in z_shapes:
            z_new = torch.randn(args.num_samples, *z) * args.temp
            z_sample.append(z_new.to(device))
    else:
        z_sample = torch.randn((args.num_samples, 3, args.img_size, args.img_size),
                                dtype=torch.float32, device=device)
        z_sample = net(z_sample, partition=True)
    
    print(f"sampling Z with size: {args.img_size}x{args.img_size}.")
    x = net(z_sample, reverse=True)
    # import ipdb; ipdb.set_trace()
    if norm_img == 'img':
        x = (x.sub(x.view(x.shape[0], -1).mean(dim=1))) #  / x.std() # (x.max() - x.min())
    elif norm_img == 'std':
        std_x = x.view(x.shape[0], -1).std(dim=1)
        x = x.div(std_x[:, None, None, None])
    elif norm_img == 'center std':
        std_x = x.view(x.shape[0], -1).std(dim=1)
        x = x.sub(x.view(x.shape[0], -1).mean(dim=1)[:, None, None, None]).div(std_x[:, None, None, None])
    elif norm_img == 'batch':
        x = (x - x.min()) / (x.max() - x.min())
    return x

def resample(z, net, device=None, args=None, norm_img = True):

    z_sample = net(z, partition=True)

    x = net(z_sample, reverse=True)
    if norm_img == 'img':
        x = (x.sub(x.view(x.shape[0], -1).mean(dim=1))) #  / x.std() # (x.max() - x.min())
    elif norm_img == 'std':
        std_x = x.view(x.shape[0], -1).std(dim=1)
        x = x.div(std_x[:, None, None, None])
    elif norm_img == 'center std':
        std_x = x.view(x.shape[0], -1).std(dim=1)
        x = x.sub(x.view(x.shape[0], -1).mean(dim=1)[:, None, None, None]).div(std_x[:, None, None, None])
    elif norm_img == 'batch':
        x = (x - x.min()) / (x.max() - x.min())
    return x


def net_noise(net, device, loc=0, scale=0.005):
    with torch.no_grad():
        for param in net.parameters():
            try:
                param.add_(torch.randn(param.size()).to(device) * scale)
            except:
                continue
    return net

class GaussianNoise(object):

    def __init__(self, mean=0., std=.1, restrict_range=True):
        self.std = std
        self.mean = mean
        self.restrict_range = restrict_range

    def __call__(self, tensor):
        tensor += torch.randn(tensor.size()) * self.std + self.mean
        if self.restrict_range:
            return tensor.clamp(1e-8, 1)
        else:
            return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)



class Normie(object):
    '''class for normies'''
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, tensor):
        tensor -= tensor.min()
        tensor /= tensor.max()
        return tensor

def find_last_epoch_model(fp):
    dirs_l = os.listdir(fp)
    dirs_e = [d for d in dirs_l if d.startswith('epoch_') 
                                     and d[-3:].isdigit()]
    dirs_e.sort()
    last_epoch = dirs_e[-1]
    print('Last model it.: ' + last_epoch)
    return fp + '/' + last_epoch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Glow model')
    parser.add_argument('--benchmark', action='store_true', help='Turn on CUDNN benchmarking')
    # parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs to train')

    # 1. Dataset : 'celeba', 'MNIST', 'CIFAR' (not tested)
    dataset_ = 'celeba'
    # 2. Architecture
    net_ = 'glow'  # 2.
    # 3. Samples dir_
    dir_ = net_ + '_' + dataset_
    # 4. GPUs
    gpus_ = '[0, 1]' # if net_ == 'densenet' and dataset_=='mnist'  else '[0]' # 4.
    # 5. resume training?
    resume_ = False # 5.
    # 6. learning_rate
    learning_rate_ = 1e-4
    # 6. resize 
    if dataset_ == 'mnist':
        in_channels_= 1
    elif dataset_ in ['celeba', 'ffhq']:
        in_channels_ = 3
        img_size_ = 128
        temp_ = 0.2
        num_sample_ = 16

    root_dir_ = 'data/' + dir_
    

    parser.add_argument('--img_size', default=img_size_, type=int, help='Image size') # changed from 1e-3 for MNIST
    parser.add_argument('--lr', default=learning_rate_, type=float, help='Learning rate') # changed from 1e-3 for MNIST
    # parser.add_argument('--resume', '-r', action='store_true', default=resume_, help='Resume from checkpoint')
    parser.add_argument('--gpu_ids', default=gpus_, type=eval, help='IDs of GPUs to use')
    parser.add_argument('--net', default=net_, help='CNN architecture (resnet or densenet)')
    parser.add_argument('--root_dir', default=root_dir_, help="Directory for storing generated samples")
    parser.add_argument('--sample_dir', default= root_dir_ +'/samples', help="Directory for storing generated samples")

    # dataset
    parser.add_argument('--dataset', '-ds', default=dataset_.lower(), type=str, help="MNIST or CIFAR-10")
    parser.add_argument('--in_channels', default=in_channels_, type=int, help='dimensionality along Channels')

    # architecture
    if net_ == 'glow':

        parser.add_argument(
        '--affine', action='store_true', help='use affine coupling instead of additive'
        )
        parser.add_argument('--no_lu', action='store_true', help="don't use LU decomposed convolution")
        parser.add_argument('--n_bits', default=5, type=int, help='number of bits')
        parser.add_argument('--n_flow', default=32, type=int, help='number of bits')
        parser.add_argument('--n_block', default=4, type=int, help='number of bits')
        parser.add_argument('--temp', default=temp_, type=float, help='temperature of sampling')
        parser.add_argument('--iter', default=200000, type=int, help='maximum iterations')
        if dataset_ == 'celeba':
            num_scales_ = 4

        elif dataset_ == 'mnist': # data/dense_test6
            batch_size_ = 1024 if len(gpus_) > 3 else 512
            ### mid_channels_ = 120
            num_samples_ = 121
            # num_scales_ = 3


    # parser.add_argument('--batch', default=batch_size_, type=int, help='Batch size')
    parser.add_argument('--num_samples', default=num_sample_, type=int, help='Number of samples at test time')
    # parser.add_argument('--num_scales', default=num_scales_, type=int, help='Real NVP multi-scale arch. recursions')

    
    best_loss = 5e5
    main(parser.parse_args())
