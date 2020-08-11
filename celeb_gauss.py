#!/var/scratch/mao540/miniconda3/envs/maip-venv/bin/python

#/var/scratch/mao540/miniconda3/envs/sobab37/bin/python

import torch 
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

import argparse
import os
import numpy as np
# from models import RealNVP, RealNVPLoss
from tqdm import tqdm
from random import randrange
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns

from scipy import stats  as distatis
from celeb_simil import * #select_model, mark_version, load_network, track_z, label_zs
# from mnist_simil import cleanup_version_f

def analyse_epoch(args, model_meta_stuff = None):
	device = torch.device("cuda:0" if torch.cuda.is_available() and len(args.gpu_ids) > 0 else "cpu")
	print("evaluating on: %s" % device)

	# select model.
	if not model_meta_stuff:
		fp_model_root, fp_model, fp_vmarker = select_model(args.root_dir, args.version, test=i)
	else:
		fp_model_root, fp_model, fp_vmarker = model_meta_stuff

	# if args.net == 'glow':
	# 	net = Glow(3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu)
	# 	net = net.to(device)
	# else:
	# 	net = RealNVP( **filter_args(args.__dict__) )
	# 	net = net.to(device)

	# mark_version(args.version, fp_vmarker) # echo '\nV-' >> fp_vmarker

	epoch = fp_model_root.split('_')[-1]
	''' stats filenames memo: '''
	stats_filename = fp_model_root + '/z_mean_std.pkl'
	if os.path.isfile(stats_filename) and not args.force and not args.tellme_y:
		print('Found cached file, skipping computations of mean and std.')
		stats = torch.load(stats_filename)
	else:
		if args.dataset == 'mnist':
			from celeb_simil import load_mnist_test 
			testloader = load_mnist_test(args)
		elif args.dataset.lower() == 'celeba':
			from celeb_simil import load_celeba_test
			testloader = load_celeba_test(args.batch, args.img_size, test=True)
		elif args.dataset == 'CIFAR-10':
			from celeb_simil import load_cifar_test 
			testloader = load_cifar_test(args)

		if not args.tellme_y: # make it explicit, it is better.
			net, loss_fn = load_network( fp_model, device, args)
			loss_fn = RealNVPLoss()
			if args.track_x:
				stats, x_cA = track_z_celeba(net, testloader, device, loss_fn, track_x=True)
				torch.save(x_cA, '{args.root_dir}/x.pkl')
			else:
				stats = track_z_celeba(net, testloader, device, loss_fn, track_x=False)
			torch.save(stats, stats_filename)

	''' Attributes annotation '''
	if args.tellme_y:
		y = tellme_ys(load_network(fp_model, device, args)[0], testloader, device)
		attr_y = Attributes().make_and_save_dataframe(y)
	else:
		attr_y = Attributes().fetch()
		'''previous tracking xz'''

	fp_distr = fp_model_root + '/distr' # distributions
	# from celeb_simil import maketree
	def lstify(s):
		if isinstance(s, str):
			return [s]
		elif isinstance(s, list):
			return s
	maketree = lambda l: [os.makedirs(p, exist_ok=True) for p in lstify(l)]
	maketree(fp_distr)
	z = stats['z'].reshape(stats['z'].shape[0], -1)
	p, ksp, m, s, b1, b2, qs = distribution_momenta(z) # for each instance. 
	# write_minmaxavg_tofile(p, ksp, args.pgaussfile, n_epoch=epoch)

	# TODO import 'x' differently
	if args.kde:
		x = torch.load(f'{args.root_dir}/x.pkl')
		x = x['x']

		# plot KDEs for each attribute
		part_length = 8
		n_segments = int(40 / part_length)
		partitioned_attributes = [[i + part_length*j for i in range(part_length)] for j in
				range(n_segments)] # == 40
		cntr = 0
		for att_selection in partitioned_attributes:
			fn = fp_distr + '/gauss_hist_{}-{}.png'.format(att_selection[0], att_selection[-1])
			plot_kde(z,x,attr_y,att_sel=att_selection,nrows=8,filename=fn,n_epoch=epoch)
			print(f'plotted {fn}')
			cntr += 1

	# mark_version(args.version, fp_vmarker, finish=True) # echo '0.2' >> fp_vmarker
	# cleanup_version_f(fp_vmarker, _removal_strings=['']) # only remove empty lines
	return epoch, p, ksp


def main(args, epochs=[160000], save=True, force=False):


	# import ipdb; ipdb.set_trace()
	p_value_counts = dict()
	for e in epochs: # [300, 400, 500, 600]:
		model_meta_stuff = select_model(args.root_dir, args.version, test=e)
		# make kde plots, return pval stats 

		fp_model_root, fp_model, fp_vmarker = model_meta_stuff
		mark_version(args.version, fp_vmarker) # echo '\nV-' >> fp_vmarker

		pcount_fn = fp_model_root + '/gauss_pvalue_counts.pkl'

		# if isinstance(kde, int):
		# 	args.kde = (e == kde) #  TO REMOVE?
		# else:
		# 	kde = args.kde

		if not os.path.isfile(pcount_fn) or force or args.kde:
			e, p, ksp = analyse_epoch(args, model_meta_stuff)
			# if not os.path.isfile(pcount_fn) or force:
			p_counts_epoch = bin_pvals_in_thresholds(p)
			p_value_counts[int(e)] = {'norm': p_counts_epoch}
			p_counts_epoch = bin_pvals_in_thresholds(ksp)
			p_value_counts[int(e)]['ks'] = p_counts_epoch
			if save:
				torch.save(p_value_counts[int(e)], pcount_fn)
		else:
			e_d = torch.load(pcount_fn)
			# one-off code. 
			# if str(list(e_d.keys())[0]) == '300':
			# 	clean_d = e_d['300']
			# 	e_d = clean_d
			# 	torch.save(e_d, pcount_fn)
			
			p_value_counts[e] = e_d
		mark_version(args.version, fp_vmarker, finish=True) # echo '0.2' >> fp_vmarker

	fn = '{}_glowcel_gautest.png'.format('e' + str(len(epochs)))
	histogram_gaussianity(p_value_counts, fn)
	fn = '{}_glowcel_ksgautest.png'.format('e' + str(len(epochs)))
	histogram_gaussianity(p_value_counts, fn, test='ks')

def bin_pvals_in_thresholds(pvals):

	thresholds = np.geomspace(1e-9, .1, num=9)
	
	counts = np.zeros(len(thresholds), dtype=np.int32)

	n_previous_th = 0
	for i, th in enumerate(thresholds):
		n_below_th = (pvals < th).sum()
		counts[i] = n_below_th - n_previous_th
		n_previous_th = n_below_th

	if pvals.shape[0] - np.sum(counts) != 0:
		counts = np.append(counts, pvals.shape[0] - np.sum(counts))
		thresholds = np.append(thresholds, 0)

	output = dict(zip(thresholds, counts))
	print(output)
	
	return output

def histogram_gaussianity(p_value_counts, filename, test='norm'):
	
	fig, axs = plt.subplots(figsize=(12, 8))

	n_thresholds = len(p_value_counts[list(p_value_counts.keys())[0]][test])
	width = .5
	hist_width = width * n_thresholds
	spacing = 1
	x_ctr = hist_width/2 + spacing/2
	x_ticks = []
	cmap = plt.cm.get_cmap('cool')

	for x, (e, pvals_dict) in enumerate(p_value_counts.items()):
		# ``explanation`` - for each specific test:
		# pvaltest_count = dict(zip(p_thresholds, p_counts))
		pvaltest_count = pvals_dict[test]
		if n_thresholds != len(pvaltest_count):
			raise ValueError('*** Seems like some p-vals are > .1!')

		x_min = x_ctr - hist_width/2 + width/2
		x_max = x_ctr + hist_width/2 - width/2
		bars_xs = np.linspace(x_min, x_max, n_thresholds)
		scaled_colorvalues = [int(v*cmap.N/n_thresholds) for v in range(n_thresholds)]
		
		# import ipdb; ipdb.set_trace()
		plt.bar(bars_xs, pvaltest_count.values(), width=width,
		        color=cmap(scaled_colorvalues)) # , label=str(e))
		# annotate
		for i, (th, cnt) in enumerate(pvaltest_count.items()):
			col_txt = 'k'
			if cnt > 8000:
				vert_align = 'top'
				if i > 1:
					col_txt = 'w'
			else:
				vert_align = 'bottom'
			plt.annotate('{}: {}'.format('p < ' + str(th), str(cnt)), c=col_txt,
			             xy=(bars_xs[i], cnt), xytext=(0, 3), rotation='90',
			             textcoords='offset points', ha='center', va=vert_align)

		x_ticks.append(x_ctr)
		x_ctr += (hist_width + spacing)

	plt.xticks(x_ticks, ['epoch ' + str(e) for e in list(p_value_counts.keys())])
	plt.savefig(filename)
	plt.close()


def write_minmaxavg_tofile(p_vals, ks_p_vals, filename='g_test.txt', n_epoch=999):
	''' to run with version specifier "V-G.0", 
	in order to avoid re-running previous analyses'''
	# check threshold, and write to file. 
	thresholds = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
	p_all = 0
	for t in thresholds:
		if np.any(p_vals > t): 
			continue
		else:
			p_all = t
	p_max = np.max(p_vals)
	p_avg = np.mean(p_vals)
	p_min = np.min(p_vals)
	
	ks_p_min = np.min(ks_p_vals)
	ks_p_avg = np.mean(ks_p_vals)
	ks_p_max = np.max(ks_p_vals)
	
	first_ = not os.path.isfile(filename)
	with open(filename, 'a') as f:
		if first_:
			f.write('{},{},{},{},{},{},{}\n'.format('epoch', 'p_avg', 'p_min', 'p_max',
			                                  'ks_p_avg', 'ks_p_min', 'ks_p_max'))
			# write headers
		f.write('{},{},{},{},{},{}\n'.format(n_epoch, p_avg, p_min, p_max,
		                                 ks_p_avg, ks_p_min, ks_p_max))


def plot_pvals(filename, log_fp='test_gAUSs.txt'):
	import pandas as pd
	fig, ax = plt.subplots(figsize=(10, 7))
	df = pd.read_csv(log_fp)
	x_ax = [i for i in range(120, 690, 10)]
	
	# import ipdb; ipdb.set_trace()
	df.plot(x='epoch', y='p_avg', kind='scatter', color='#cb4b16', ax=ax,
	        logy=True, rot=5, fontsize=8, label='p-value')
	df.plot(x='epoch', y='ks_p_avg', kind='scatter', color='#2aa198', ax=ax,

	        logy=True, label='KS p-value' )

	ax.set_xticks(range(120, 690, 10), minor=True)
	ax.grid(which='minor', alpha=0.3)
	ax.grid(which='major', alpha=0.5)
	plt.ylabel("$p$ under H0")
	plt.title("Gaussianity test")
	plt.savefig(filename)
	plt.close()
	
	
def distribution_momenta(z, quantiles=[0.1,0.25, 0.5, 0.75,0.9], mean=False, idx=None,
	                       ks=True):
	if mean and idx is not None:
		raise ValueError('Please only mean or idx.')

	# last_dim = len(z.shape) - 1
	_, p_vals = distatis.normaltest(z, axis=1)
	# being non-parametric, can be used later with different z\s
	# kolmogorov smirnov takes time: flag it. 
	if ks:
		_, ks_p_vals = ndim_kstest(z, axis=1)

	
	mus = np.mean(z, axis=1)
	sigmas = np.std(z, axis=1)
	k_vals = distatis.kurtosis(z, axis=1)
	s_vals = distatis.skew(z, axis=1)

	# import ipdb; ipdb.set_trace()
	quartiles  = np.quantile(z, quantiles, axis=1, overwrite_input=True)

	if ks:
		return (p_vals,ks_p_vals,mus,sigmas,s_vals,k_vals,quartiles)
	else:
		return (p_vals,mus,sigmas,s_vals,k_vals,quartiles)





def ndim_kstest(z, axis=1, cdf=None, args=()):
	if not cdf:
		normal_distr = distatis.norm(loc=0, scale=1)
		cdf = normal_distr.cdf
	z = np.apply_along_axis(distatis.kstest, axis, z, cdf)
	return (z[:,0], z[:,1])

def plot_kde(z, x,att, att_sel=range(10), nrows=3, filename='sampled_densities.png', n_epoch=99):

	ncols = len(att_sel) # 
	fig, axs = plt.subplots(nrows=nrows, ncols=ncols, 
			                    sharex='all', sharey='all', figsize=(ncols * 3, nrows+1))
	digit = 0

	for col in range(len(att_sel)):
		# z_att = z[att.iloc[:, col].astype(bool)]
		att_idcs = np.argwhere(np.array(att.iloc[:, att_sel[col] ]))
		att_idcs = np.random.choice(att_idcs.flatten(), size=nrows, replace=False)
		z_subset = z[att_idcs]
		x_subset = x[att_idcs]
		# random_idcs= [np.random.choice(digit_idx.flatten(), replace=False)
		# 		                                              for i in range(n_rows)]
		# use random idcs to feed z[idcs] to `distribution_momenta(...)`
		p_vals, mus, sigmas, s_vals, k_vals, qs = distribution_momenta(z_subset, ks=False)

		for row in range(nrows):
			sns.distplot(z_subset[row], bins=20, rug=False, kde_kws=dict(linewidth=0.8),
			                            color='C'+str(digit), ax=axs[row, col])
			# axs[row, col].text(-4, .4,
			#                    '$\mu$: {:.2f}\n'
			#                    '$\sigma$: {:.2f}\n'.format(
			#                    mus[row], sigmas[row]),
			axs[row, col].set_xlabel('$\mu$: {:.2f}, $\sigma$: {:.2f}\np-val: {:.5f}'.format(
			                         mus[row], sigmas[row], p_vals[row]),
			                   # '$\\beta_1$: {:.3f}\n'
			                   # '$\\beta_2$: {:.3f}' s_vals[row], k_vals[row]),
			                         fontsize=7) #, linespacing=0.75)
			if row == 0:
				axs[row, col].set_title(att.columns[ att_sel[col] ] ) # nrows * col + row])

			maxy_dist = np.max(z_subset[row])
			miny, maxy = axs[row, col].get_ylim()
			minx, maxx = axs[row, col].get_xlim()

			# show handwritten sign-digit for each distribution. 
			x_img = np.moveaxis(x_subset[row].reshape(3, 64, 64), 0, -1)
			x_imagebox = OffsetImage(x_img, zoom=0.5)
			x_imagebox.image.axes = axs[row, col]
			x_ab = AnnotationBbox(x_imagebox, xy=(1, 0.4), 
					          xycoords='data', boxcoords=("axes fraction", "data"))
			axs[row, col].add_artist(x_ab)

			# if row == 0:
			# 	axs[row, col].title.set_text(f'Digit {digit + 1}')

			for q in qs[:,row]:
				axs[row, col].axvline(x=q, ymin=0,ymax=maxy_dist, c='red', alpha=0.3, linewidth=0.5)


		digit += 1
	# plt.tight_layout()
	plt.setp(axs, xlim=(-6, 6)) # ylim seems to be fine. 
	plt.subplots_adjust(wspace=0.5, hspace=0.5, bottom=0.15)
	plt.savefig(filename)
	plt.close()


def track_xz(net, loader, device, loss_fn, **kwargs):

	net.eval()
	
	with tqdm(total=len(loader.dataset)) as progress:

		digits_stds = [np.array([]).reshape(0, 1) for i in range(10)]
		digits_means = [np.array([]).reshape(0, 1) for i in range(10)]

		''' not quite but useful for reference. '''
		digits_z = [np.array([]).reshape(0, 1, 28, 28) for i in range(10)]
		digits_x = [np.array([]).reshape(0, 1, 28, 28) for i in range(10)]

		for x, y in loader:
			x = x.to(device)
			z, sldj = net(x, reverse=False)
			mean_z = z.mean(dim=[1, 2, 3]) # `1` is channel.
			std_z = z.std(dim=[1, 2, 3])

			for n in range(10):
				# check if concatenation axis needs to be specified for ZZZZ.
				digit_indices = (y==n).nonzero()
				digit_indices_1dim = (y==n).nonzero(as_tuple=True)
				dig_std_z = std_z[ digit_indices ]
				dig_mean_z = mean_z[ digit_indices ]
				z_dig = z[digit_indices_1dim]
				x_dig = x[digit_indices_1dim]
				# here we concatenate each 'array' to the previous one (or initialized arr. of size=(0, 1)).
				digits_stds[n] = np.concatenate(( digits_stds[n], dig_std_z.to('cpu').detach().numpy() ))
				digits_means[n] = np.concatenate(( digits_means[n], dig_mean_z.to('cpu').detach().numpy() ))
				# concatenate whole z space. 
				digits_z[n] = np.concatenate(( digits_z[n], z_dig.to('cpu').detach().numpy() ))
				digits_x[n] = np.concatenate(( digits_x[n], x_dig.to('cpu').detach().numpy() ))

			loss = loss_fn(z, sldj)

			progress.update(x.size(0))
	# add loss.avg() and bpd.avg() somewhere in the plot. 
	
	return {'std': digits_stds, 'mean': digits_means,
	        'z': digits_z, 'x': digits_x}

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='RealNVP on CIFAR-10')

	# parser.add_argument('--num_samples', default=121, type=int, help='Number of samples at test time')

	# XXX PARAMS XXX
	version_ = 'V-G.1'
	force_ = False
	# General architecture parameters
	dataset_ = 'celeba'
	net_ = 'glow'
	gpus_ = '[0]'
	force_ = False
	ys_ = False
	track_x = False
	kde_ = True
	pgaussfile_ = 'gceleb_gauss.txt' # for testing purposes
	# Analysis


	in_channels_ = 3
	img_size_ = 64
	if net_ == 'glow':
		if dataset_ == 'celeba':
			batch_size_ = 32
			root_dir_ = 'data/glow_celeba'
			num_samples_ = 64
			if gpus_ == '[0, 1]':
				batch_size_ = 128
				num_samples_ = 128
		parser.add_argument(
		'--affine', action='store_true', help='use affine coupling instead of additive'
		)
		parser.add_argument('--no_lu', action='store_true', help="don't use LU decomposed convolution")
		parser.add_argument('--n_bits', default=5, type=int, help='number of bits')
		parser.add_argument('--n_flow', default=32, type=int, help='number of bits')
		parser.add_argument('--n_block', default=4, type=int, help='number of bits')
		parser.add_argument('--temp', default=0.7, type=float, help='temperature of sampling')
	else: raise NotImplementedError

	parser.add_argument('--track_x', action='store_true', default=track_x)
	parser.add_argument('--kde', action='store_true', default=kde_)
	parser.add_argument('--tellme_y', action='store_true', default=ys_)
	parser.add_argument('--force', '-f', action='store_true', default=force_, help='Re-run z-space analyses.')
	parser.add_argument('--version', '-v', default=version_, type=str, help='Analyses iteration')

	parser.add_argument('--net', default=net_, help='CNN architecture (resnet or densenet)')
	parser.add_argument('--batch', default=batch_size_, type=int, help='Batch size')
	parser.add_argument('--root_dir', default=root_dir_, help='Analyses root directory.')
	parser.add_argument('--dataset', '-ds', default=dataset_, type=str, help='MNIST or CIFAR-10')
	parser.add_argument('--in_channels', default=in_channels_, type=int, help='dimensionality along Channels')
	parser.add_argument('--img_size', default=img_size_, type=eval)
	# glow
	parser.add_argument('--pgaussfile', default=pgaussfile_, type=str, help='log gaussianity to file')
	parser.add_argument('--gpu_ids', default=gpus_, type=eval, help='IDs of GPUs to use')
	parser.add_argument('--benchmark', action='store_true', help='Turn on CUDNN benchmarking')
	parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader threads')
	
	main(parser.parse_args())
