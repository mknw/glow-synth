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


def main(args):

	device = torch.device("cuda:0" if torch.cuda.is_available() and len(args.gpu_ids) > 0 else "cpu")
	print("training on: %s" % device)
	start_epoch = 0


	if args.net == 'glow':
		model = Glow(3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu)
		net = model.to(device)
	else:
		net = RealNVP( **filter_args(args.__dict__) )
		net = net.to(device)

	if str(device).startswith('cuda'):
		net = torch.nn.DataParallel(net, args.gpu_ids)
		cudnn.benchmark = args.benchmark

	if args.resume: # or not args.resume:
		# Load checkpoint.
		args.model_dir = find_last_model_relpath(args.root_dir) # /model_{str(i + 1).zfill(6)}.pt'
		print(f'Resuming from checkpoint at {args.model_dir}')
		checkpoint = torch.load(args.model_dir+'/model.pth.tar')
		net.load_state_dict(checkpoint['net'])
		global best_loss
		best_loss = checkpoint['test_loss']
		# we start epoch after the saved one (avoids overwrites).
		start_epoch = checkpoint['epoch'] + 1
	else:
		os.makedirs(args.root_dir, exist_ok=True)
		os.makedirs(args.sample_dir, exist_ok=True)

	if args.net == 'glow':
		optimizer = optim.Adam(net.parameters(), lr=args.lr)
	else:
		loss_fn = RealNVPLoss()
		param_groups = util.get_param_groups(net, args.weight_decay, norm_suffix='weight_g')
		optimizer = optim.Adam(param_groups, lr=args.lr, eps=1e-7)

	train(args, net, device, optimizer, start_epoch)


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

def train(args, net, device, optimizer, start_epoch):

	dataset = iter(sample_celeba(args.batch, args.img_size))
	n_bins = 2. ** args.n_bits

	z_sample = []
	import ipdb; ipdb.set_trace()
	z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
	for z in z_shapes:
		z_new = torch.randn(args.num_sample, *z) * args.temp
		z_sample.append(z_new.to(device))
	loss_meter = AverageMeter()
	bpd_meter = AverageMeter()
	# import ipdb; ipdb.set_trace()
	p_imgs = 0
	with tqdm(range(start_epoch, args.iter)) as pbar:
		for i in pbar:
			x, _ = next(dataset)
			x = x.to(device)

			if i == 0:
				with torch.no_grad():
					# log_p, logdet, _ = net.module(x + torch.rand_like(x) / n_bins)
					log_p, logdet, _ = net(x + torch.rand_like(x) / n_bins)
					continue
			else:
				log_p, logdet, _ = net(x + torch.rand_like(x) / n_bins)

			logdet = logdet.mean()

			loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
			net.zero_grad()
			loss.backward()
			# warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
			warmup_lr = args.lr
			optimizer.param_groups[0]['lr'] = warmup_lr
			optimizer.step()
			loss_meter.update(loss.item(), x.size(0))
			bpd_meter.update(bits_per_dim(x, loss_meter.avg))

			pbar.set_description(
					f'Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}; imgs: {p_imgs}'
			)
			p_imgs += x.size(0)
			if i % 1000 == 0:
				# SAMPLE
				with torch.no_grad():
					torchvision.utils.save_image(
						net(z_sample, reverse=True).cpu().data,
						args.sample_dir + f'/{str(i).zfill(6)}.png',
						normalize=True,
						nrow=10,
						range=(-0.5, 0.5),
					)
				with open(f'{args.root_dir}/log', 'a') as l:
					report = f'{loss.item():.5f},{log_p.item():.5f},{log_det.item():.5f},{warmup_lr:.7f},{p_imgs}\n'
					# print("Writing to disk: " + report + ">> {}/log".format(args.root_dir))
					l.write(report)


			if i % 10000 == 0:
				# TEST
				model_dir = f'{args.root_dir}/epoch_{str(i).zfill(6)}'
				os.makedirs(model_dir)
				torch.save({'net': net.state_dict(),
				            'test_loss': loss_meter.avg,
										'epoch': i 
										},
										f'{model_dir}/model.pth.tar'
				)
				torch.save(
					optimizer.state_dict(), f'{model_dir}/optim_{str(i + 1).zfill(6)}.pt'
				)

def sample_celeba(batch, image_size, test=False):
	if not test:
		split = 'train'
		transform = transforms.Compose([
			transforms.CenterCrop(160),
			transforms.Resize(size=image_size),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
		])
	else:
		split = 'test'
		transform = transforms.Compose([
		transforms.CenterCrop(160),
		transforms.Resize(size=image_size),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
	])
	target_type = ['attr', 'bbox', 'landmarks']
	dataset = datasets.CelebA(root='data', split=split, target_type=target_type[0], download=True, transform=transform)
	loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=8)
	loader = iter(loader)

	while True:
		try:
			yield next(loader)
		except StopIteration:
			loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=8)
			loader = iter(loader)
			yield next(loader)


def calc_z_shapes(n_channel, input_size, n_flow, n_block):
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
	x = torch.sigmoid(x)
	return x, z


def test(epoch, net, testloader, device, loss_fn, **args):
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
	save_dir = args['dir_samples'] + '/epoch_{:03d}'.format(epoch) #  + str(epoch)
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

	# import ipdb; ipdb.set_trace()
	sample_fields = ['num_samples', 'in_channels', 'resize_hw']
	images, latent_z = sample(net, device=device, **filter_args( args, fields=sample_fields ) )

	# plot x and z
	num_samples = args['num_samples']
	images_concat = torchvision.utils.make_grid(images, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
	z_concat = torchvision.utils.make_grid(latent_z, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
	torchvision.utils.save_image(images_concat, save_dir+'/x.png')
	torchvision.utils.save_image(z_concat, save_dir+'/z.png')

	# with open(, 'wb') as z_serialize:
	# 	pickle.dump(latent_z, z_serialize)
	torch.save(latent_z, f = save_dir+'/z.pkl')

	# dict keys as returned by "train"
	train_loss = args['train_loss']
	train_bpd = args['train_bpd']
	report = [epoch, loss_meter.avg, bpd_meter.avg] + [train_loss, train_bpd]

	dir_samples = args['dir_samples']
	with open('{}/log'.format(dir_samples), 'a') as l:
		report = ", ".join([str(m) for m in report])
		report += "\n"
		print("\nWriting to disk:\n" + report + "At {}".format(dir_samples))
		l.write(report)


def filter_args(arg_dict, fields=None):
	"""only pass to network architecture relevant fields."""
	if not fields:
		# if arg_dict['net_type'] == 'resnet':
		fields = ['net_type', 'num_scales', 'in_channels', 'mid_channels'] 
		# elif arch['net_type'] == 'densenet':
		# 	arch_fields = ['net_type', 'num_scales', 'in_channels', 'mid_channels', 'depth']
	return {k:arg_dict[k] for k in fields if k in arg_dict}


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
	dirs_l = os.listdir(fp)
	dirs_e = [d for d in dirs_l if d.startswith('epoch_')]
	dirs_e.sort()
	last_epoch_model = dirs_e[-1]
	return fp + '/' + last_epoch_model

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
	gpus_ = '[0, 1]' if net_ == 'densenet' and dataset_=='mnist'  else '[0]' # 4.
	# 5. resume training?
	resume_ = False # 5.
	# 6. learning_rate
	learning_rate_ = 1e-4
	# 6. resize 
	if dataset_ == 'mnist':
		in_channels_= 1
	elif dataset_ == 'celeba':
		in_channels_= 3
		img_size_ = 64

	root_dir_ = 'data/' + dir_
	

	parser.add_argument('--img_size', default=img_size_, type=int, help='Image size') # changed from 1e-3 for MNIST
	parser.add_argument('--lr', default=learning_rate_, type=float, help='Learning rate') # changed from 1e-3 for MNIST
	parser.add_argument('--resume', '-r', action='store_true', default=resume_, help='Resume from checkpoint')
	parser.add_argument('--gpu_ids', default=gpus_, type=eval, help='IDs of GPUs to use')
	parser.add_argument('--net', default=net_, help='CNN architecture (resnet or densenet)')
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
		parser.add_argument('--temp', default=0.7, type=float, help='temperature of sampling')
		parser.add_argument('--iter', default=200000, type=int, help='maximum iterations')
		if dataset_ == 'celeba':
			num_scales_ = 4
			batch_size_ = 32
			num_sample_ = 32

		elif dataset_ == 'mnist': # data/dense_test6
			batch_size_ = 1024 if len(gpus_) > 3 else 512
			### mid_channels_ = 120
			num_samples_ = 121
			# num_scales_ = 3


	parser.add_argument('--batch', default=batch_size_, type=int, help='Batch size')
	parser.add_argument('--root_dir', default=root_dir_, help="Directory for storing generated samples")
	parser.add_argument('--num_sample', default=num_sample_, type=int, help='Number of samples at test time')
	parser.add_argument('--num_scales', default=num_scales_, type=int, help='Real NVP multi-scale arch. recursions')

	
	best_loss = 5e5
	main(parser.parse_args())
