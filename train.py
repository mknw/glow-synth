#!/var/scratch/mao540/miniconda3/envs/maip-venv/bin/python

import argparse
import os
import torch 
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

from numpy import log
# from models import RealNVP, RealNVPLoss
from model import Glow
from tqdm import tqdm
# from train_r import calc_z_shapes
from shell_util import AverageMeter, bits_per_dim
# from load_data import load_network

from config.config import ConfWrap
from utils import ModelNotFoundError

def main(C):

    device = torch.device("cuda:0" if torch.cuda.is_available() and len(C.net.gpus) > 0 else "cpu")
    print("training on: %s" % device)
    start_epoch = 0


    # net, = load_network(model_fp, device, C.net)
    model = Glow(3, C.net.n_flows, C.net.n_blocks, affine=C.net.affine, conv_lu=C.net.lu_conv)
    net = model.to(device)
    if str(device).startswith('cuda'):
        net = torch.nn.DataParallel(net, C.net.gpus)
        cudnn.benchmark = C.training.benchmark

    if C.resume: # or not C.resume:
        # Load checkpoint. TODO: uncomment last line, delete the next one

        # import ipdb; ipdb.set_trace()
        C.model_dir = find_last_model_relpath(C.training.root_dir) # /model_{str(i + 1).zfill(6)}.pt'
        # C.model_dir = f'{C.root_dir}/epoch_180000'
        print(f'Resuming from checkpoint at {C.model_dir}')
        checkpoint = torch.load(C.model_dir+'/model.pth.tar')
        net.load_state_dict(checkpoint['net'])
        global best_loss
        best_loss = checkpoint['test_loss']
        # we start epoch after the saved one (avoids overwrites).
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch: {checkpoint['epoch']}")
    else:
        os.makedirs(C.training.root_dir, exist_ok=True)
        os.makedirs(C.training.sample_dir, exist_ok=True)

    optimizer = optim.Adam(net.parameters(), lr=float(C.training.learning_rate))
    if C.resume:
        try:
            optimizer.load_state_dict(torch.load(f'{C.model_dir}/optim.pt'))
        except KeyError:
            print(f'error loading {C.model_dir}/optim.pt')
            optimizer.load_state_dict(torch.load('data/glow_meyes_2/epoch_310000/optim.pt'))

    z_sample = find_or_make_z(C.training.root_dir + '/z_samples.pkl',
                              3, C.training.img_size, C.net.n_flows, C.net.n_blocks,
                              C.training.n_samples, C.training.temp, device)
    train(C.training, net, device, optimizer, start_epoch, z_sample)


def calc_loss(log_p, logdet, image_size, n_bins):
    # log_p = calc_log_p([z_list])
    n_pixel = image_size * image_size * 3

    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )

def train(config, net, device, optimizer, start_epoch, z_sample):

    if config.dataset == 'celeba':
        from load_data import sample_celeba
        dataset = iter(sample_celeba(config.batch_size, config.img_size))
    elif config.dataset == 'ffhq':
        from load_data import sample_from_directory
        if config.img_size > 128:
            dataset = iter(sample_from_directory('data/FFHQ/images1024x1024', config.batch_size, config.img_size))
        else:
            dataset = iter(sample_from_directory('data/FFHQ/thumbnails128x128', config.batch_size, config.img_size))
    elif config.dataset == 'meyes':
        from load_data import sample_FFHQ_eyes
        from load_data import RandomRotatedResizedCrop as RRRC
        dataset = iter(sample_FFHQ_eyes(config.batch_size, config.img_size, shuffle=True,
                                        transform=RRRC(output_size=config.img_size)))

    n_bins = 2. ** config.n_bits


    loss_meter = AverageMeter()
    bpd_meter = AverageMeter()

    p_imgs = 0
    net.train()
    pbar = tqdm(range(start_epoch, config.iter))
    pbar.update(start_epoch); pbar.refresh()
    for i in pbar:
        x, _ = next(dataset)
        x = x.to(device)

        if i == 0:
            with torch.no_grad():
                log_p, logdet, _ = net(x + torch.rand_like(x) / n_bins)
                continue
        else:
            log_p, logdet, _ = net(x + torch.rand_like(x) / n_bins)

        logdet = logdet.mean()

        loss, log_p, log_det = calc_loss(log_p, logdet, config.img_size, n_bins)
        net.zero_grad()
        loss.backward()
        # warmup_lr = C.lr * min(1, i * batch_size / (50000 * 10))
        warmup_lr = config.learning_rate
        optimizer.param_groups[0]['lr'] = warmup_lr
        optimizer.step()
        loss_meter.update(loss.item(), x.size(0))
        bpd_meter.update(bits_per_dim(x, loss_meter.avg))

        pbar.set_description(
                f'Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}; imgs: {p_imgs}'
        )
        p_imgs += x.size(0)
        if i % 1000 == 0:
            # save model
            if i % 10000 == 0:
                # TEST
                model_dir = f'{config.root_dir}/epoch_{str(i).zfill(6)}'
                os.makedirs(model_dir, exist_ok=True)
            else:
                model_dir = config.root_dir
            torch.save({'net': net.state_dict(),
                        'test_loss': loss_meter.avg,
                                    'epoch': i 
                                    },
                                    f'{model_dir}/model.pth.tar'
            )
            torch.save(
                optimizer.state_dict(), f'{model_dir}/optim.pt'
            )
            # Generate new samples.
            with torch.no_grad():
                torchvision.utils.save_image(net(z_sample, reverse=True).cpu().data,
                                             f'{config.sample_dir}/{str(i).zfill(6)}.png',
                                             normalize=True,
                                             nrow = int(config.n_samples ** 0.5),
                                             range=(-0.5, 0.5))

            with open(f'{config.root_dir}/log', 'a') as l:
                report = f'{loss.item():.5f},{log_p.item():.5f},{log_det.item():.5f},{warmup_lr:.7f},{p_imgs}\n'
                # print("Writing to disk: " + report + ">> {}/log".format(config.root_dir))
                l.write(report)

        net.train()



def find_or_make_z(path, C, img_size, n_flows, n_block, num_sample, t, device):

    if os.path.isfile(path):
        z_sample = torch.load(path)

    else:
        z_sample = []
        z_shapes = calc_z_shapes(3, img_size, n_flows, n_block)
        for z in z_shapes:
            z_new = torch.randn(num_sample, *z) * t
            z_sample.append(z_new.to(device))

        torch.save(z_sample, path)
    return z_sample

def calc_z_shapes(n_channel, input_size, n_flows, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2
        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))
    return z_shapes


def save_imgrid(tensor, name):
    grid = torchvision.utils.make_grid(tensor, nrow=int(tensor.shape[0] ** 0.5), padding=1, pad_value=255)
    torchvision.utils.save_image(grid, name)
    return

def sample(net, num_samples, in_channels, device, resize_hw=None):
    """Sample from RealNVP model.

    Args:
        net (torch.nn.DataParallel): The RealNVP model wrapped in DataParallel.
        batch_size (int): Number of samples to generate.
        device (torch.device): Device to use.
    """
    
    if not resize_hw:
        side_size = 28
    else:
        side_size, side_size = resize_hw
    print(f"sampling with z space sized: {side_size}x{side_size}.")
    z = torch.randn((num_samples, in_channels, side_size, side_size), dtype=torch.float32, device=device) #changed 3 -> 1
    x, _ = net(z, reverse=True)
    return x, z


def test(epoch, net, testloader, device, loss_fn, **C):
    global best_loss
    net.eval()
    loss_meter = util.AverageMeter()
    bpd_meter = util.AverageMeter()
    with torch.no_grad():
        with tqdm(total=len(testloader.dataset)+1) as progress_bar:
            for x, _ in testloader:
                x = x.to(device)
                z, sldj = net(x, reverse=False)
                loss = loss_fn(z, sldj)
                loss_meter.update(loss.item(), x.size(0))
                # bits per dimensions
                bpd_meter.update(util.bits_per_dim(x, loss_meter.avg), x.size(0))

                progress_bar.set_postfix(loss=loss_meter.avg,
                                                                 bpd=bpd_meter.avg)
                progress_bar.update(x.size(0))
                
    # Save checkpoint
    save_dir = C['dir_samples'] + '/epoch_{:03d}'.format(epoch) #  + str(epoch)
    os.makedirs(save_dir, exist_ok=True)

    # if loss_meter.avg < best_loss or epoch % 10 == 0 or
    # 		epoch > 100 or epoch < 20:
    if True:
        print('\nSaving...')
        state = {
            'net': net.state_dict(),
            'test_loss': loss_meter.avg,
            'epoch': epoch,
        }
        torch.save(state, save_dir + '/model.pth.tar')
        best_loss = loss_meter.avg

    sample_fields = ['num_samples', 'in_channels', 'resize_hw']
    images, latent_z = sample(net, device=device, **filter_args( C, fields=sample_fields ) )

    # plot x and z
    num_samples = C['num_samples']
    images_concat = torchvision.utils.make_grid(images, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
    z_concat = torchvision.utils.make_grid(latent_z, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
    torchvision.utils.save_image(images_concat, save_dir+'/x.png')
    torchvision.utils.save_image(z_concat, save_dir+'/z.png')

    # with open(, 'wb') as z_serialize:
    # 	pickle.dump(latent_z, z_serialize)
    torch.save(latent_z, f = save_dir+'/z.pkl')

    # dict keys as returned by "train"
    train_loss = C['train_loss']
    train_bpd = C['train_bpd']
    report = [epoch, loss_meter.avg, bpd_meter.avg] + [train_loss, train_bpd]

    dir_samples = C['dir_samples']
    with open('{}/log'.format(dir_samples), 'a') as l:
        report = ", ".join([str(m) for m in report])
        report += "\n"
        print("\nWriting to disk:\n" + report + "At {}".format(dir_samples))
        l.write(report)


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

def find_last_model_relpath(fp):
    ''' Select epoch checkpoint or intermediate backup.
    The euristic used in the filepath name of samples
    args: root directory'''
    # TODO: implement model loading + extraction of saved epoch from dictionary 
    # in order to validate whether the model is the most up to date.

    dirs_l = os.listdir(fp)
    samples_l = os.listdir(fp + '/samples')
    dirs_e = [int(d.split('_')[-1]) for d in dirs_l if d.startswith('epoch_')]
    samples_fns = [int(png.split('.')[0]) for png in samples_l if png.endswith('png')]
    dirs_e.sort()
    samples_fns.sort()
    dirs_e, samples_fns = [l if len(l) > 0 else [-1] for l in [dirs_e, samples_fns]]
    # Should stay `>=`
    if dirs_e[-1] >= samples_fns[-1]:
        if dirs_e[-1] == -1 and samples_fns[-1] == -1:
            # no model was saved.
            raise ModelNotFoundError(f'no model checkpoint found in {fp}.')
        # checkpoint @ epoch
        out = f'{fp}/epoch_{dirs_e[-1]}'
    else:
        # output intermediate model dir.
        # (backup for samples epochs with (n % 10000 != 0))
        out = fp
    return out



if __name__ == '__main__':

    C = ConfWrap(fn='config/ffhq256lu_c.yml')
    # C = ConfWrap(fn='config/ffhq128_c.yml')
    C.resume = True
    C.training.sample_dir = C.training.root_dir + '/samples'
    main(C)

