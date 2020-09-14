#!/var/scratch/mao540/miniconda3/envs/maip-venv/bin/python
# maybe TODO: change venv to: 'sobab/bin/python' ^

import torch 
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.nn.functional import normalize

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from umap import UMAP
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import BoundaryNorm as BoundaryNorm

# import util
import argparse
import os
import shutil
from random import randrange
# from mnist_simil import cleanup_version_f

def main(args, epochs=[160000, 150000], save=True, force=False):

    p_value_counts = dict()
    for e in epochs:
        model_meta_stuff = select_model(args.root_dir, args.version, test=e)
        fp_model_root, fp_model, fp_vmarker = model_meta_stuff
        # mark_version(args.version, fp_vmarker) # echo '\nV-' >> fp_vmarker

        # if not os.path.isfile(pcount_fn) or force or args.kde:
        analyse_epoch(args, model_meta_stuff)
        # mark_version(args.version, fp_vmarker, finish=True) # echo '0.2' >> fp_vmarker


def analyse_epoch(args, model_meta_stuff = None):
    device = torch.device("cuda:0" if torch.cuda.is_available() and len(args.gpu_ids) > 0 else "cpu")
    print("evaluating on: %s" % device)
    
    # import ipdb; ipdb.set_trace()

    # select model.
    if not model_meta_stuff:
        fp_model_root, fp_model, fp_vmarker = select_model(args.root_dir, args.version, test=i)
    else:
        fp_model_root, fp_model, fp_vmarker = model_meta_stuff

    epoch = fp_model_root.split('_')[-1]
    # stats
    stats_filename = fp_model_root + '/z_mean_std.pkl'
    if os.path.isfile(stats_filename) and not args.force and not args.track_x:
        #             make sure tracking isn't asked ^^^^^
        print('Found cached file, skipping computations of mean and std.')
        stats = torch.load(stats_filename)
    else:
        if args.dataset == 'mnist':
            testloader = load_mnist_test(args)
        elif args.dataset.lower() == 'celeba':
            testloader = load_celeba(args.batch, args.img_size, test=True)
        elif args.dataset == 'CIFAR-10':
            testloader = load_cifar_test(args)

        net, loss_fn = load_network( fp_model, device, args)
        net.eval()
        if args.track_x:
            stats, x_cA = track_z_celeba(net, testloader, device, loss_fn, track_x=True)
            torch.save(x_cA, f'{args.root_dir}/x.pkl')
        else:
            stats = track_z_celeba(net, testloader, device, loss_fn, track_x=False)
        torch.save(stats, stats_filename)

    ''' Attributes annotation '''
    if args.tellme_y:
        if 'net' not in dir():
            net, _ = load_network( fp_model, device, args)
            net.eval()
        y = tellme_ys(load_network(fp_model, device, args), testloader, device)
        attr_y = Attributes().make_and_save_dataframe(y)
    else:
        attr_y = Attributes().fetch()

    ''' filepaths '''
    fp_distr = fp_model_root + '/distr' # distributions
    fp_simil = fp_model_root + '/similarity' # similarity
    fp_replace = fp_model_root + '/replace_from_grandz'
    fp_pca = fp_model_root + '/pca'
    fp_umap = fp_model_root + '/umap'
    paths = [fp_distr, fp_simil, fp_replace, fp_pca]
    # lstify = lambda s: [s] if isinstance(s, str)
    def lstify(s):
        if isinstance(s, str):
            return [s]
        elif isinstance(s, list):
            return s
    maketree = lambda l: [os.makedirs(p, exist_ok=True) for p in lstify(l)]
    maketree(paths)
    # [os.makedirs(ppp, exist_ok=True) for ppp in [fp_distr, fp_simil, fp_replace, fp_pca]]
    
    
    ''' distributions analyses '''
    # print("analysing z distribution... ")
    # scatter_all(stats, fp_distr + '/meanstds.png', epoch)

    # scatter_attr(stats, attr_y, fp_distr + '/faces_subplots3.png', epoch)
    # # # # violin_eachdigit(stats, fp_distr + '/dig_violins.png', epoch)
    # # ''' distance analysis '''
    # distances = calculate_distance(stats, attr_y, measure='mean')
    # heatmap(distances, attr_y, fp_simil + '/distances_mean.png',
    # 			plot_title='Average distance between attributes in Z space (means)')
    # distances = calculate_distance(stats, attr_y, measure='std')
    # heatmap(distances, attr_y, fp_simil + '/distances_std.png',
    # 			plot_title='Average distance between digits in Z space (std)')
    # distances = calculate_distance(stats, attr_y, joint=True)
    # heatmap(distances, attr_y, fp_simil + '/distances.png')

    # # distances = y_distance_z(stats)
    # # measure = 'mean'
    # # for m in range(distances.shape[0]):
    # # 	heatmap(distances[m], fp_simil + f'/pixelwise_dist_{measure}.png',
    # # 							 plot_title=f'pixelwise {measure} similarity')
    # # 	measure = 'std'

    # all_zs = grand_z(stats, attr_y)
    # # overall Z for each digit (averaged across batches).
    # plot_grand_z(all_zs, attr_y.columns, fp_model_root + '/grand_zs_normimg.png')
    # plot_grand_z(all_zs, attr_y.columns, fp_model_root + '/grand_zs_normarray.png', norm='array')
    # plot_grand_z(all_zs, attr_y.columns, fp_model_root + '/grand_zs_normarrayb.png', norm='arrayb')
    # # plot_grand_z_rgb(all_zs, attr_y.columns, fp_model_root + '/grand_zs_rgb.png')
    # if 'net' not in dir():
    # 	net, _ = load_network( fp_model, device, args)
    # 	net.eval()

    # 
    # # maketree(fp_replace)
    # with torch.no_grad():
    # 	for t in [.75, 0.9, 1.]:
    # 		for i in range(1, 11):
    # 			k = int(12288 / 2 ** i)
    # 			sample_from_crafted_z(net, all_zs, absolute=True, kept=k, device=device, reps=1,
    # 								 save_dir=fp_replace, temp=t) #monster_mode=True)

    # 		sample_from_crafted_z(net, all_zs, absolute=True, kept=12288, device=device, reps=1,
    # 								 save_dir=fp_replace, temp=t) #monster_mode=True)

    # 		# sample_from_crafted_z(net, all_zs, absolute=True, kept=12288, device=device, reps=1,
    # 		# 						 save_dir=fp_replace, temp=t, monster_mode=True)


    # ''' dimensionality reduction '''
    # if 'net' not in dir():
    # 	net, _ = load_network( fp_model, device, args)
    # 	net.eval()

    # dataset = stats['z'].reshape(stats['z'].shape[0], -1)

    # # # # pick_components = 350
    # fp_pca += '/nosig'; maketree([fp_pca])
    # for pick_components in [50, 100, 1000]:
    # 	print('computing PCA...{}...'.format(pick_components), end='')
    # 	pca = PCA(n_components=pick_components).fit(dataset) # dataset.data
    # 	if pick_components == 'mle':
    # 		print(f' N PCs found: {pca.n_components_}')
    # 	analyse_principal_components(pca,stats,attr_y,fp_pca,36, net, device)

    print('umap analysis')
    if 'net' not in dir():
        net, _ = load_network( fp_model, device, args)
        net.eval()
    fp_umap += '/resamp_temp8'; maketree(fp_umap)
    umap_inverse_wrapper(stats, attr_y, fp_umap, net, device)
    print('done')

    # # UMAP -- `test_umap` will use directories `<fp_model_root>/umap/{,3d}`
    # fn_prefix = fp_model_root + '/umap'
    # os.makedirs(fn_prefix+'/3d', exist_ok=True)
    # for nn in [7, 10, 20, 100, 400]:
    # 	# add option to reduce nn for 3d. 
    # 	dims = [2, 3] if nn in [7, 20] else [2]
    # 	for md in [0.01, 0.1, 0.5, 0.9]:
    # 		for d in dims:
    # 			test_umap(stats, fn_prefix, n_neighbors=nn, min_dist=md*0.05, n_components=d)


def issue_z_from_pc(PC, stats, filename):
    raise NotImplementedError
    z_s = grand_z(stats)
    z_s = z_s.reshape(z_s.shape[0], -1)
    components = PC['components']
    for i in range(10):
        
        print('h'+i)
        # every N
        for c in components:
            # every PC
            pass
    pass


def PCA_eig_np(Z, k, center=True, scale=False):
    '''
    https://medium.com/@ravikalia/pca-done-from-scratch-with-python-2b5eb2790bfc
    '''
    n, p = Z.shape
    ones = np.ones([n, 1])
    # subtract from each column its mean, to ensure mean = 0.
    h = ((1/n) * np.matmul(ones, ones.T)) if center else np.zeros([n, n])
    H = np.eye(n) - h
    Z_center = np.matmul(H, Z)
    covariance = 1/(n-1) * np.matmul(Z_center.T, Z_center)
    # divide each column by its std. Only if outcome is independent of variance.
    scaling = np.sqrt(1/np.diag(covariance)) if scale else np.ones(p)
    scaled_covariance = np.matmul(np.diag(scaling), covariance)
    w, v = np.linalg.eig(scaled_covariance)
    components = v[:, :k]
    explained_variance = w[:k]
    return {'z': Z, 'k': k, 'components': components.T,
                'exp_var': explained_variance}


def analyse_principal_components(pca, stats, att, fp_pca,pk, net=None, device=None):
    '''
    Arguments: 
        - pca: sklearn.decomposition. PCA object. after calling .fit()
         stats: Z stats
        - fp_pca: root filepath for plot saving.
        - pk: componets to show in plots PCgrid.
    '''

    print("plotting reconstructed Z... ", end='')
    sel_attr = [[i + 5*j for i in range(5)] for j in range(8)] # == 40
    for att_selection in sel_attr:
        fn = fp_pca + '/rZ_att-{}_ncomps-{}.png'.format(''.join([str(i) for i in att_selection]),
                                                      pca.n_components_)
        plot_rZ(pca, stats['z'], att, filename=fn, n_examples=6,
               net=net, device=device, selected_attributes=att_selection)
    print("done.")

    fn = fp_pca + '/PCgrid_first-{}.png'.format(pk)
    if not os.path.isfile(fn):
        print("plotting components in grid format... ", end='')
        plot_PCgrid(pca, fn)
        print("done.")

    print("plotting variance explained... ", end='')
    fn = fp_pca + '/VE_ncomps-{}.png'
    plot_expvar(pca.n_components_, pca.explained_variance_ratio_, fn)
    print("done.")

    os.makedirs(fp_pca+'/red', exist_ok=True)
    for i in range(40):
        print(f"plotting {att.columns[i]} reduced z's... ", end='')
        fn = fp_pca + '/red/rZatt_{}-{}.png'.format(i, att.columns[i])
        plot_reduced_dataset(pca, stats['z'], att, k=10, att_ind=i, filename=fn)
    print("done.")


def plot_rZ(pca, z_s, att, filename, n_examples, net, device, selected_attributes=5):

    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    '''z_s, y'''
    if not isinstance(selected_attributes, int):
        nrows, ncols = n_examples, len(selected_attributes)# 3 instances, 10 n of sel. categories.
    else:
        nrows, ncols = n_examples, selected_attributes
        selected_attributes = np.random.choice([i for i in range(40)], size=ncols, replace=False)

    h_size = n_examples
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*6,nrows*2))

    pc_i = 1
    net.eval()
    for col in range(ncols):
        with torch.no_grad():
            # ori_Z = z_s[dig_idx].astype(np.float32)
            # Select nrows of celebrities
            n_att = selected_attributes[col]
            ori_Z = z_s[att.iloc[:, n_att].astype(bool)].astype(np.float32)
            celeb_idx = np.random.randint(ori_Z.shape[0], size=nrows)
            ori_Z = ori_Z[celeb_idx].copy()
            
            ### original Zs
            # keep original Zs for plotting; oriZ for generation
            # variable without underscores `_` are used for the GPU (pytorch).
            ori_Z = ori_Z.reshape(nrows,3,64,64)
            oriZ = torch.from_numpy(ori_Z).to(device)
            oriZ = net(oriZ, partition=True)
            oriX = net(oriZ, reverse=True, resample=True)
            oriX = oriX.cpu().detach().numpy()
            # Transform with var. explained by PCs
            red_Z = pca.transform(ori_Z.reshape(nrows, -1))
            rec_Z = pca.inverse_transform(red_Z)
            ### reconstruced Zs
            rec_Z = rec_Z.reshape(nrows, 3, 64, 64)
            # keep rec_Z for plotting; recZ for generation.
            recZ = torch.from_numpy(rec_Z.astype(np.float32)).to(device)
            recZ = net(recZ, partition=True)
            recX = net(recZ, reverse=True, resample=True)
            recX = recX.cpu().detach().numpy()
            ### normalize over array cel_rZ
            cel_rZ = (rec_Z - rec_Z.min()) / (rec_Z.max() - rec_Z.min())
            # axs[0, col].set_title(f"{col}")
            axs[0, col].set_title(att.columns[n_att], fontsize='small')

        for row in range(nrows):
            
            axs[row, col].imshow(np.moveaxis(cel_rZ[row], 0, -1))

            axs[row, col].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False,
                    right=False, left=False, labelleft=False)

            # oriX_imbox.image.axes = axs[row, col]
            annotext = "$O\;\mu: {:.2f}, \sigma: {:.2f} || R\;\mu: {:.2f}, \sigma:{:.2f}$".format(
                    ori_Z[row].mean(), ori_Z[row].std(), rec_Z[row].mean(), rec_Z[row].std())
            axs[row, col].set_xlabel(annotext, fontsize='xx-small')

            # Show original and reconstructed X
            before = np.moveaxis(oriX[row].reshape(3,64,64), 0, -1)
            before = (before - before.min()) / (before.max() - before.min())
            oriX_imbox = OffsetImage(before, zoom=1.2)
            oX_ab = AnnotationBbox(oriX_imbox, xy=(-0.6, 0.5), 
                              xycoords='data', boxcoords="axes fraction")
            axs[row, col].add_artist(oX_ab)

            after = np.moveaxis(recX[row].reshape(3,64,64), 0, -1)
            after = (after - after.min()) / (after.max() - after.min())
            recX_imbox = OffsetImage(after, zoom=1.2)
            # x_imagebox.image.axes = axs[row, col]
            rX_ab = AnnotationBbox(recX_imbox, xy=(1.6, 0.5), 
                              xycoords='data', boxcoords="axes fraction")
            axs[row, col].add_artist(rX_ab)


    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_PCgrid(PCA, filename, pk=None, reconstruct=False):
    if not pk: # Plot Komponents
        pk = min(PCA.n_components_, 7*7)
    PCs = PCA.components_
    n_pcs = PCA.n_components_
    var_exp = PCA.explained_variance_
    # y = PCs['y']
    nrows = ncols = int(pk ** 0.5)
    # nrows +=1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*2, nrows*2))
    PCs = PCs.reshape(n_pcs, 3, 64, 64)
    # PCs = (comp - comp.min()) / (comp.max() - comp.min())
    # cmap = plt.cm.RdBu
    mean_red_z = PCA.mean_.reshape(3, 64, 64)
    mean_red_z = (mean_red_z - mean_red_z.min()) / (mean_red_z.max() - mean_red_z.min())
    axs[0, 0].imshow(np.moveaxis(mean_red_z, 0, -1))
    axs[0, 0].set_title('$\mu$')
    pc_i = 0
    for row in range(nrows):
        for col in range(ncols):
            if col == 0 and row == 0: continue # mean is displayed here.

            if pc_i < pk: # for the case where pca.n_PCs < 25
                comp_img = (np.moveaxis(PCs[pc_i].copy(), 0, -1) - PCs[pc_i].min()) / (PCs[pc_i].max() - PCs[pc_i].min())
                axs[row, col].imshow(comp_img)
                axs[row, col].set_title('PC{}'.format(pc_i+1))
            # else:  # if pc_i >= pk:
            # 	# axs[row, col].remove()
            # 	# axs[row, col] = None
            pc_i += 1


    plt.tight_layout()
    fig.subplots_adjust(top=.88)
    plt.savefig(filename) # TOFIX
    plt.close()


def PCA_test(z_s, k, center=False, mode='self'):
    
    dataset, y = label_zs(z_s)
    n_datapoints = dataset.shape[0]
    p = dataset.shape[1] # feature number

    if mode == 'sklearn':
        PCs = PCA(n_components=k).fit(dataset) # dataset.data
        return {'z': dataset, 'k': k, 'components': PCs.components_,
                    'exp_var': PCs.explained_variance_, 'y': y}
    elif mode == 'self':
        PCs = PCA_eig_np(dataset, k, center)
        PCs['y'] = y
        return PCs

def select_att_series(attr_df, att_ind, max_attributes=10):

    if not isinstance(att_ind, (int, list, tuple)):
        raise TypeError


    if isinstance(att_ind, int):
        subsel_df = pd.concat([attr_df.iloc[:, att_ind], (attr_df.iloc[:, att_ind] == 0)], axis=1)
        overall_indexer = (subsel_df == 1).any(axis=1)
        return pd.concat([subsel_df, overall_indexer], axis=1)
    else:
        length = len(att_ind)
        if length > 1: # assumed sequence
            selection = attr_df.iloc[:, np.array(att_ind)]
            attribute_names = list(selection.columns)
            complement_categs_names = ['comp_' + c for c in attribute_names]
            # concatenate selected attributes df with its complementary.
            subsel_df = pd.concat([selection, (attr_df.iloc[:, np.array(att_ind)] == 0)], axis=1)
            subsel_df.columns = attribute_names + complement_categs_names
            overall_indexer = (subsel_df == 1).any(axis=1)
            
            return pd.concat([subsel_df, overall_indexer], axis=1)

        elif length < 1 or length > max_attributes:
            raise ValueError

    


def plot_reduced_dataset(pca, z_s, att, k, att_ind, filename):
    # sort from highest variance
    if isinstance(pca, PCA):
        # for att_s in sel_att:
        # 	if att_s not in att.columns:
        # 		raise ValueError(f'Expected attribute name, got {att_s}')

        components = pca.components_[:k][::-1]
        var_exp = pca.explained_variance_[:k][::-1]
        ratio_var_exp = pca.explained_variance_ratio_[:k][::-1]
        '''z_s, y = label_zs(z_s)'''
        
        sel_att_df = select_att_series(att, att_ind)
        red_z = pca.transform(z_s[ sel_att_df.iloc[:, -1]].reshape(sel_att_df.shape[0], -1))
        reduced_z = red_z[:, :k][:,::-1]   # PCs['X'].T becomes (306,10000)
    else: # should be type: sklearn.decomposition.PCA
        raise NotImplementedError

    symbols = "." # can be used for orthogonal attributes.
    n_pcs = k # can use this to index components and create grid.
    fs = int(n_pcs * 2)
    fig, axs = plt.subplots(n_pcs, n_pcs, figsize=(fs, fs), sharex='col', sharey='row')
    cmap = plt.cm.winter # get_cmap('Set1')

    # Use subset dataframe turn 1 hot vectors into indices,
    # then add column for "both" categories if overlapping.
    color_series, overlapping_attributes = category_from_onehot(sel_att_df)
    # color_series += 2 # make it red
    
    for row in range(n_pcs):
        # randomly permutation of reduced datapoints for 
        # visualization that is evened among categories. 
        # indices = np.random.permutation(reduced_z.shape[0])
        # reduced_z = np.take(reduced_z, indices, axis=0)
        # y = np.take(y, indices)
        for col in range(n_pcs):
            if row > col:
                path_c = axs[row, col].scatter(reduced_z[:,col], reduced_z[:,row], c=np.array(color_series), cmap=cmap, s=.50, alpha=0.6)
                axs[row, col].annotate('% VE:\nC{}={:.2f}\nC{}={:.2f}'.format(n_pcs - row, ratio_var_exp[row]*100,
                                         n_pcs-col, ratio_var_exp[col]*100), xy=(0.7, 0.7), xycoords='axes fraction', fontsize='xx-small')
                if row == n_pcs-1:
                    axs[row, col].set_xlabel(f'component {n_pcs-col}') 
                    axs[row, col].tick_params(axis='x', reset=True, labelsize='x-small')
                if col == 0:
                    axs[row, col].set_ylabel(f'component {n_pcs-row}')
                    axs[row, col].tick_params(axis='y', reset=True, labelsize='x-small')
            else:
                axs[row, col].remove()
                axs[row, col] = None

    handles, labels = path_c.legend_elements(prop='colors')
    if overlapping_attributes:
        assert isinstance(att_ind, (list, tuple))
        labels = att.columns[np.array(att_ind)] + ['both']
    else:
        assert isinstance(att_ind, int)
        labels= [att.columns[att_ind]] + ['Complement cat.']
    plt.legend(handles, labels, bbox_to_anchor=(.75, .75), loc="upper right", 
               bbox_transform=fig.transFigure)
    
    fig.tight_layout()
    fig.subplots_adjust(top=.88)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()



def plot_PCA(PCs, filename):
    # sort from highest variance
    if isinstance(PCs, dict):
        components = PCs['components'][::-1]
        var_exp = PCs['exp_var'][::-1]
        z_s = PCs['z']
        y = PCs['y']
        reduced_z = components.dot(z_s.T) # might want to check this.
    else: # should be type: sklearn.decomposition.PCA
        components = PCs.components_[::-1]
        var_exp = PCs.explained_variance_[::-1]
        reduced_Z = PCs.transform()

    n_pcs = len(var_exp) # can use this to index components and create grid.
    fig, axs = plt.subplots(n_pcs, n_pcs, figsize=(12, 12), sharex='col', sharey='row')
    
    for row in range(n_pcs):
        for col in range(n_pcs):
            if row > col:
                axs[row, col].scatter(components[row], components[col], s=.50)# label=f'{col}x{row}')
                axs[row, col].annotate('var.exp.:\nC{}={:.3f}\nC{}={:.3f}'.format(n_pcs - row, var_exp[row],
                                         n_pcs-col, var_exp[col]), xy=(.30, .30), fontsize='xx-small')
                if row == n_pcs-1:
                    axs[row, col].set_xlabel(f'component {n_pcs-col}') 
                    axs[row, col].tick_params(axis='x', reset=True, labelsize='x-small', which='both')
                if col == 0:
                    axs[row, col].set_ylabel(f'component {n_pcs-row}')
                    axs[row, col].tick_params(axis='y', reset=True, labelsize='x-small', which='both')
            else:
                axs[row, col].remove()
                axs[row, col] = None
    
    fig.tight_layout()
    fig.subplots_adjust(top=.88)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

#	horrible i know
#	nrows = ncols = len(var_exp) # can use this to index components and create grid.
#	fig, axs = plt.subplots(nrows, ncols, figsize=(12, 12), sharex='col', sharey='row')
#	
#	for row in range(nrows):
#		for col in range(ncols):
#			if row > col:
#				axs[row, col].scatter(components[row], components[col], s=.50)# label=f'{col}x{row}')
#				axs[row, col].annotate('var.exp.:\nC{}={:.3f}\nC{}={:.3f}'.format(5 - row, var_exp[row],
#					                     5-col, var_exp[col]), xy=(.30, .30), fontsize='xx-small')
#				if row == nrows-1:
#					axs[row, col].set_xlabel(f'component {5-col}') 
#				if col == 0:
#					axs[row, col].set_ylabel(f'component {5-row}')
#				if row == nrows-1 or col == 0:
#					axs[row, col].tick_params(reset=True, labelsize='x-small')
#			else:
#				axs[row, col].remove()
#				axs[row, col] = None
    

def plot_expvar(n_pcs, r_var_exp, filename):
    
    fig, ax = plt.subplots(figsize=(8, 6))

    # n_pcs = len(var_exp)
    cs = np.cumsum(r_var_exp)
    ax.plot(cs)

    ax.set_ylabel("cumulative ratio of explained variance")

    ax.axhline(cs[-1], color="k", alpha=0.5)

    trans = mpl.transforms.blended_transform_factory(
              ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(0, cs[-1], "{:.4f}".format(cs[-1]), color="blue", transform=trans,
              ha="right", va="center")

    ax.set_xlim(left=0, right=n_pcs)

    xtick_vals, xtick_labels = list(plt.xticks())
    if float(n_pcs) not in xtick_vals:
        plt.xticks(xtick_vals + [n_pcs], xtick_labels + [n_pcs])
    
    # plt.title("variance explained first PC")
    plt.savefig(filename.format(n_pcs))
    plt.close()


def make_interpolated_grid(angles=None):
    '''
    make grid.
    angles should be: 
        - (xmin, ymin, xmax, ymax) of type int.
    '''
    
    # xmin, ymin, xmax, ymax = angles
    # angles = [[xmin, ymax], [xmax, ymax], [xmin, ymin], [xmax, ymin]]

    corners = np.array(angles)
    test_pts = np.array([
        (corners[0]*(1-x) + corners[1]*x)*(1-y) +
        (corners[2]*(1-x) + corners[3]*x)*y
        for y in np.linspace(0, 1, 10)
        for x in np.linspace(0,1,10)
    ])
    return test_pts


def rotate_around_point(point, radians, origin=(0, 0)):
    """Rotate a point around a origin.
    [taken from https://gist.github.com/LyleScott/e36e08bfb23b1f87af68c9051f985302]
    I call this the "low performance" version since it's recalculating
    the same values more than once [cos(radians), sin(radians), x-ox, y-oy).
    It's more readable than the next function, though.
    """
    x, y = point
    ox, oy = origin

    qx = ox + np.cos(radians) * (x - ox) + np.sin(radians) * (y - oy)
    qy = oy + -np.sin(radians) * (x - ox) + np.cos(radians) * (y - oy)
    return qx, qy


def translation_matrix(xy, to_origin=True):
    transl_mtx= np.eye(3)
    if to_origin:
        transl_mtx[:-1, -1] = -1 * xy # grid back inplace.
    else:
        transl_mtx[:-1, -1] = xy # grid back inplace.
    return transl_mtx

def scale_matrix(xy_scale_factors):
    x_sf, y_sf = xy_scale_factors
    return np.array([[x_sf,0,0],[0,y_sf,0], [0,0,1]]) # only scale x, rest is rotation.

def rotation_matrix(angle):
    theta = angle
    return np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0],[0,0,1]])

def make_grid(boundaries, degrees=0, scale_factor = 0.5, edge_scale_factors=(0.3, 1)):
    xmin, ymin, xmax, ymax = boundaries
    # get center of grid.
    radians = degrees * np.pi / 180 # TODO define degrees
    grid_center = np.array((xmin + (xmax-xmin) / 2, ymin + (ymax-ymin) / 2))
    # make array for affine transf. out of upper/lower/left/right boundaries.
    vertices = np.array([[xmin, ymax], [xmax, ymax], [xmin, ymin], [xmax, ymin]])
    vertices = np.concatenate([vertices, np.ones(vertices.shape[0]).reshape(vertices.shape[0],1)], axis=1)
    # translation matrix: to origin and back to grid centre (or point cloud center).
    transl_mtx_to_origin = translation_matrix(grid_center)
    transl_mtx_to_grid_center = translation_matrix(grid_center, to_origin=False)
    # define scaling matrix (for upper edge first, whole matrix then)
    edge_scale_mtx = scale_matrix(edge_scale_factors)
    scale_mtx = scale_matrix((scale_factor, scale_factor))
    # define rotation matrix
    rot_mtx = rotation_matrix(radians)

    # Apply transformations.
    alt_mtx = transl_mtx_to_origin.dot(vertices.T)
    # only change upper edge, the rest is integral to the whole frame
    alt_mtx[:,:2] = edge_scale_mtx.dot(alt_mtx[:, :2])
    alt_mtx = scale_mtx.dot(alt_mtx)
    alt_mtx = rot_mtx.dot(alt_mtx)
    alt_mtx = transl_mtx_to_grid_center.dot(alt_mtx).T
    # produce smooth linear grid among vertices.
    test_pts = make_interpolated_grid(alt_mtx)
    return  test_pts


def category_from_onehot(onehot_df, idcs=(0, 1), vals=(1, 2), over_val=None):
    if idcs == (0, 1) and onehot_df.shape[1] > 3:
        print("Warning: dataframe is misshaped for idcs (0, 1). Continue at own risk.")
    i, j = idcs
    val_i, val_j = vals
    color_series = onehot_df.iloc[:, i].copy()
    color_series[color_series == 1] = val_i
    color_series[onehot_df.iloc[:, j]] = val_j

    if (onehot_df.iloc[:, np.array(idcs)].sum(axis=1) > 1).any():
        overlapping_onehot_ = True
        print('W: overlapping categories found: filling with value: {over_val}')
        color_series[onehot_df.iloc[:,np.array(idcs)].sum(axis=1)] = over_val
    else: overlapping_onehot_ = False
    return color_series, overlapping_onehot_



def umap_inverse_wrapper(stats, att, fp_umap, net=None, device=None, n_neighbors_l=None):

    dataset = stats['z'].reshape(stats['z'].shape[0], -1)

    ''' attributes '''
    # att_ind = (20, 31)
    '''here''' # and call it in plot_inverse_umap
    
    if not n_neighbors_l:
        n_neighbors_l = [20, 100, 400]
        min_dist_l = [0.2, 0.2, 0.2]
    else:
        n_neighbors_l = [n_neighbors_l]
    for nn , md in zip(n_neighbors_l, min_dist_l):
        print(f'UMAP: nearest_n = {nn}, min_dist = .2')
        deg = 45
        knn = 100
        knn_w = 'distance'
        # grid_s = 0.35
        umap = UMAP(n_neighbors=nn, min_dist= md,
                                n_components=2, random_state=42)
        mapper = umap.fit(dataset)
        inverse_sampling = True
        choice, choice2 = np.random.randint(40, size=2)
        for a_i in [choice]: # range(40):
            for a_j in [choice2]: # range(a_i+1, 40):

                attributes = select_att_series(att, att_ind=(a_i, a_j))
                col_arr = attributes.iloc[:, 0]
                sym_arr = attributes.iloc[:, 1]
                names = attributes.iloc[:,0].name + '-' + attributes.iloc[:,1].name

                print(f'computing inverse umap sampling for: {names}', end='')
        
                basename = '/{}-{}_{}_nn{:d}_md{:.2f}_knn{:d}'.format(a_i, a_j, names, nn,md,knn)

                fn = fp_umap + basename + '.jpg'
                plot_inverse_umap(fn, att=att, n_neighbors=nn, min_dist=md, knn=knn, deg_rot=deg,
                                  col_arr=col_arr, sym_arr=sym_arr, knn_weights=knn_w,
                                  inverse_sampling=inverse_sampling,mapper=mapper,net=net,
                                  device=device, grid_scale=1, temp=0.8)

                inverse_sampling = False
                break


def plot_knn_boundaries(x1, x2, y, nn=5, weights='distance', h=.02, ax=None,
                          n_colors=2, cmap=None):

    from sklearn import neighbors
    from matplotlib.colors import ListedColormap
    # Lighter tab10 version:
    if not cmap:
        tab20 = mpl.cm.get_cmap('tab20')
        newcolors = tab20(np.linspace(0, 1, 20))[1 : 1+n_colors*2 : 2]
        cmap = ListedColormap(newcolors)

    clf = neighbors.KNeighborsClassifier(n_neighbors=nn, weights=weights)
    dataset = np.c_[x1, x2]
    clf.fit(dataset, y.astype(int))

    x_min, x_max = x1.min() - 1, x1.max() + 1
    y_min, y_max = x2.min() - 1, x2.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    if not ax:
        plt.pcolormesh(xx, yy, Z, cmap='tab10', alpha=.2)
    else:
        ax.pcolormesh(xx, yy, Z, cmap=cmap)
        return ax, clf


def plot_inverse_umap(filename, stats=None, att=None,
                          n_neighbors=None, min_dist=0.02, n_components=2, inverse_sampling=False,
                          deg_rot=45, mapper=None, col_arr=None, sym_arr=None, knn=100,
                          knn_weights=None, net=None, device=None, grid_scale=0.7, temp=1,**kwargs):

    if not mapper:
        raise NotImplementedError
        if not stats:
            raise ValueError
        dataset = stats['z'].reshape(stats['z'].shape[0], -1)
        print('computing UMAP projection: ', end='')
        print(f'n_neighbors = {n_neighbors}; min_dist = {min_dist}...', end='')
        umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                    n_components=n_components,
                    metric='euclidean', random_state=42)
        mapper = umap.fit(dataset)

    grid_boundaries = [mapper.embedding_[:,0].min(), mapper.embedding_[:,1].min(),
              mapper.embedding_[:,0].max(), mapper.embedding_[:,1].max()]
    test_pts = make_grid(grid_boundaries, degrees=deg_rot, scale_factor=grid_scale)

    # setup plotting grid
    from matplotlib.gridspec import GridSpec
    if inverse_sampling:
        fig = plt.figure(figsize=(22, 11))
        gs = GridSpec(10, 20, fig)
        scatter_ax = fig.add_subplot(gs[:, :10])
        faces_axes = np.zeros((10, 10), dtype=object)
        for i in range(10):
            for j in range(10):
                faces_axes[i, j] = fig.add_subplot(gs[i, 10+j])
    else:
        fig = plt.figure(figsize=(10, 10))
        scatter_ax = fig.add_subplot()

    # colormaps for knn boundaries and UMAP scatter.
    from matplotlib.colors import ListedColormap

    n_colors = col_arr.nunique() + sym_arr.nunique()

    pastel_cmap = mpl.cm.get_cmap('Pastel1')
    newcolors_knn = pastel_cmap(np.linspace(0, 1, 8))[0 : n_colors]
    pastel_subcmap_knn = ListedColormap(newcolors_knn)

    set1_cmap = mpl.cm.get_cmap('Set1')
    newcolors_a = set1_cmap(np.linspace(0, 1, 8))[0 : n_colors//2]
    set1_subcmap_a = ListedColormap(newcolors_a)
    newcolors_b = set1_cmap(np.linspace(0, 1, 8))[n_colors//2: n_colors]
    set1_subcmap_b = ListedColormap(newcolors_b)

    col_arr_knn = col_arr.copy() # 0:  F / no S  || 1: M / no S

    col_arr_knn[ (col_arr == 0) & (sym_arr == 0)] = 0 # F / no S
    col_arr_knn[ (col_arr == 1) & (sym_arr == 0)] = 1 # M / no S
    col_arr_knn[ (col_arr == 0) & (sym_arr == 1)] = 2 # F / S
    col_arr_knn[ (col_arr == 1) & (sym_arr == 1)] = 3 # M / S 
    # KNNeighbors
    scatter_ax, knn_clf = plot_knn_boundaries(mapper.embedding_[:,0], mapper.embedding_[:,1],
                                     weights=knn_weights, y=col_arr_knn, nn=knn,
                                     ax=scatter_ax, cmap=pastel_subcmap_knn)
    # scatter embeddings. 
    # We plot 2 categories and their respective complementaries.
    # Category_1 is colorcoded (in col_arr)
    # Category_2 is represented by markers (in sym_arr)
    # here we separate the dataset by markers (symbols).
    embedding_cat_2b = mapper.embedding_[sym_arr.astype(bool), :] # Smiling
    embedding_cat_2a = mapper.embedding_[(sym_arr== 0), :]       #  Not smiling
    # col_arr = np.array(col_arr)
    col_arr_cat_2a = col_arr[(sym_arr==0)]          # no Smiling
    col_arr_cat_2b = col_arr[sym_arr.astype(bool)] # Smiling
    

    # No Smiling (F/M)# red / blue
    sctt_ax_cat_2a = scatter_ax.scatter(embedding_cat_2a[:,0], embedding_cat_2a[:,1],
                     c=col_arr_cat_2a.astype(int), marker = 'x', cmap=set1_subcmap_a, s=7)
    # Smiling (F/M)# green / violet
    sctt_ax_cat_2b = scatter_ax.scatter(embedding_cat_2b[:,0], embedding_cat_2b[:,1],
                     c=col_arr_cat_2b.astype(int), marker = '+', cmap=set1_subcmap_b, s=7)

    # scatter_ax.set(xticks=[], yticks=[])
    if inverse_sampling:
        scatter_ax.scatter(test_pts[:,0], test_pts[:,1], marker='x', c='k', s=15, alpha=1)
        print(f'starting inverse transform for {test_pts.shape[0]} datapoints...', end='')

        inv_transformed_points = mapper.inverse_transform(test_pts[:,:-1])
        print('done!')
        # ### original Zs
        # # keep original Zs for plotting; oriZ for generation
        # torch.cuda.empty_cache()
        tZ = torch.from_numpy(inv_transformed_points.reshape(100, 3,64,64).astype(np.float32)).to(device)
        # tZ = torch.randn((100, 3, 64, 64), dtype=torch.float32, device=device) #changed 3 -> 1

        tX = np.array([], dtype=np.float32).reshape(0, 3, 64, 64)
        for t_Z_ in tZ.view(4, 25, 3, 64, 64):
            with torch.no_grad():
                # TODO: add temperature
                t_x_ = net(net(t_Z_ * temp, partition=True), reverse=True, resample=True)
                tX = np.concatenate([tX, t_x_.to('cpu').detach().numpy()])

        # plot generated digits:
        for i in range(10):
            for j in range(10):
                x_face = tX[i*10+j].reshape(3, 64, 64)
                x_face = (x_face - x_face.min()) / (x_face.max() - x_face.min())
                faces_axes[i, j].imshow(np.moveaxis(x_face, 0, -1))
                faces_axes[i, j].set(xticks=[], yticks=[])
    
    handles_2a, labels_2a = sctt_ax_cat_2a.legend_elements(prop='colors')
    handles_2b, labels_2b = sctt_ax_cat_2b.legend_elements(prop='colors')

    handles = handles_2a + handles_2b
    color_attributes = 'no-' + col_arr.name, col_arr.name
    marker_attributes =  'no-' + sym_arr.name, sym_arr.name

    labels = ['/'.join([c, m]) for c in marker_attributes for m in color_attributes]

    scatter_ax.legend(handles, labels, loc='best')

    scatter_ax.legend(handles, labels, loc='best')
    # plt.title('n_neighbors = {:d}; min_dist = {:.2f}'.format(n_neighbors, min_dist))

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(' Saved {}'.format(filename.split('/')[-1]))


def test_umap(stats, fn_prefix, n_neighbors=15, min_dist=0.1, n_components=2,
                  metric='euclidean', **kwargs):

    # dataset, col_arr = label_zs(stats['z'])
    dataset = stats['z'].reshape(stats['z'].shape[0], -1)

    print('computing UMAP projection: ', end='')
    print(f'n_neighbors = {n_neighbors}; min_dist = {min_dist}...', end='')

    reductor = UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                                    n_components=n_components, 
                                    metric=metric, random_state=42)
    embeddings = reductor.fit_transform(dataset)

    if n_components == 2:
        fig, ax = plt.subplots(figsize=(10, 10))
        scatter = ax.scatter(embeddings[:,0], embeddings[:,1], c=col_arr, cmap='Spectral', s=4)
    elif n_components == 3:
        fn_prefix += '/3d'
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(11, 11))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(embeddings[:,0], embeddings[:,1], embeddings[:,2], alpha=0.3,
                                                c=col_arr, cmap='Spectral', s=4)
    
    handles, labels = scatter.legend_elements(prop='colors')
    ax.legend(handles, labels, loc='best', title='off-line digits')
    plt.title(f'n_neighbors = {n_neighbors}; min_dist = {min_dist}')

    filename = fn_prefix + '/nn{:d}.jpg'.format(n_neighbors, min_dist, n_components)
    plt.savefig(filename)
    plt.close()
    print(' Saved {}'.format(filename.split('/')[-1]))



def test_arrays():
    '''deprecated utility function'''
    arr_1 = np.arange(24).reshape(2,3,4)
    arr1 = arr_1[0,::2].copy()
    arr2 = arr_1[1,1:].copy()
    arr_1[0,::2] = arr2[:,::-1]
    arr_1[1,1:] = arr1[:,::-1]
    arr_3= np.zeros(shape=(arr_1.shape[0], 1, 3, 4))
    arr_2 = np.zeros(shape=(arr_1.shape[0], 3, 4))
    arr_2[:] = 99
    return (arr_1, arr_2, arr_3)

def replace_highest_along_axis(arr_ref, arr_2, arr_3, k):
    '''
    Replaces values from arr_2 to arr_3, according to highest k values
    in arr_ref.
    Input:
        - arr_ref: reference array to look up highest k values.
        - arr_2: source array, to extract values from.
        - arr_3: destination array, to inject values to at given indices.
    Output:
        - arr_3 with vals from arr_2 at indices presenting arr_1 k-highest values. 
    '''
    assert (arr_ref.shape[0] % arr_2.shape[0]) == 0, "Arrays mismatch"
    # arr_ref, arr_2, arr_3 = test_arrays()
    shape_arr_3 = arr_3.shape
    shape_arr_2 = arr_2.shape
    arr_mask = np.zeros_like(arr_2)
    reps = arr_3.shape[0] // arr_2.shape[0] # repetitions.
    # Reference array, with flattened batches.
    arr_ref_fb = arr_ref.reshape(arr_ref.shape[0], -1)
    # select k highest values. 
    maxk_b_ind = np.argpartition(arr_ref_fb, -k)[:, -k:] # without first :?
    maxk = np.take_along_axis(arr_2.reshape(arr_2.shape[0], -1), maxk_b_ind, -1)
    # put maxk in arr_mask for plotting
    np.put_along_axis(arr_mask.reshape(arr_2.shape[0], -1), maxk_b_ind, maxk, -1)
    if reps > 1:
        maxk = maxk.repeat(reps, axis=0)
        maxk_b_ind = maxk_b_ind.repeat(reps, axis=0)

    # arr_3 = arr_3.reshape(arr_ref_fb.shape, -1)
    np.put_along_axis(arr_3.reshape(arr_3.shape[0], -1), maxk_b_ind, maxk, -1) # channel dimension
    return arr_mask.reshape(shape_arr_2), maxk, arr_3.reshape(shape_arr_3)


def replace_highest(arr_1, arr_2, arr_3, k=1):
    '''
    Deprecated.
    Replaces values from arr_2 to arr_3, according to highest k values
    in arr_1.
    Input:
        - arr_1: reference array to look up highest k values.
        - arr_2: source array, to extract values from.
        - arr_3: destination array, to inject values at given indices.
    Output:
        - arr_3 with vals from arr_2 at indices presenting arr_1 k-highest values. 
    '''
    # arr_1, arr_2, arr_3 = test_arrays()
    assert arr_1.size == arr_3.size, "Arrays mismatch"
    # 1. reshape with shape=(batch_size, H*W)
    arr_b = arr_1.reshape(arr_1.shape[0], -1)
    # 2. find indices for k highest values for each item along 1st dimension. 
    maxk_b_ind= np.argpartition(arr_b, -k)[:, -k:]

    # 3. flatten and unravel indices
    maxk_ind_flat = maxk_b_ind.flatten() #<- LET OP: unravelling flattened inds.
    maxk_ind_shape = np.unravel_index(maxk_ind_flat, arr_1.shape)
    # unravel: form indices coordinates system references to: (arr_1.shape)
    batch_indices = np.repeat(np.arange(arr_1.shape[0]), k) # (batch_size, k).
    maxk_indices = tuple([batch_indices] + [ind for ind in maxk_ind_shape])

    maxk = arr_2.reshape(arr_3.shape)[maxk_indices] # 3. resume this. 
    arr_3[maxk_indices] = maxk
    return maxk_indices, maxk, arr_3

def compute_delta(grand_zs, absolute=True):
    diff = np.zeros_like(grand_zs)

    for i in range(grand_zs.shape[0]):
        # i
        if absolute:
            diff[i] = np.abs(grand_zs[i] - np.mean(np.delete(grand_zs, i, axis=0), axis=0))
        else:
            diff[i] = grand_zs[i] - np.mean(np.delete(grand_zs, i, axis=0), axis=0)
    return diff

def craft_z(grand_zs, absolute, kept=None, reps=10, fold=False, device="cuda:0"):
    ''' Create z's from average, but with gaussian noise.
    Inputs:
        - nd_array: number-digits array with grand average of all z's spaces.
        - kept: integer or float. If integer, equals number of pixels to be kept
                                  if float, and fold=True, equals proportion of 
                                  pixels to be kept.
        - fold: dictates whether kept is read as absolute pixels count, or proportion.
    Outputs:
        - Artificial Z's for each digit. 
    '''
    # mean = np.mean(grand_zs, axis=0) # should be ~= 0.
    # abs_diff = np.abs(grand_zs - mean)
    diff = compute_delta(grand_zs, absolute=absolute)

    batch_size = grand_zs.shape[0] * reps
    batch = torch.randn((batch_size, 3, 64, 64), dtype=torch.float32, device='cpu').numpy() # TODO: CHANGE 'CPU'
    arr_mask, _, batch = replace_highest_along_axis(diff, grand_zs, batch.copy(), kept)
    del diff
    return arr_mask, batch


def sample_from_crafted_z(net, all_zs, absolute, kept, reps, device, save_dir, temp=.75, monster_mode=False):
    ''' 
    Input:
        all_zs: n-dimensional but also faces grand Z array.
    Output: plot.
    '''
    mask_zs, z = craft_z(all_zs, absolute=absolute, kept=kept, reps=reps)
    if monster_mode:
        B, C, H, W = z.shape
        hw = int(H * (B ** 0.5))
        z = z.reshape(1, C, hw, hw)
        # tile mask_zs horizontally (obtain 28 x 280 batches)
        mask_zs = np.tile(mask_zs, reps=(1, 1, 10))
        mask_zs.reshape(3, 640, 640)
        kept = f'{kept}x{reps}'

    z = torch.from_numpy(z*temp).to(device)
    z_p = net(z, partition=True)
    x = net(z_p, reverse=True)
    
    x = x.to('cpu').detach().numpy()

    plot_grand_z(x, Attributes().headers, save_dir + f'/k{kept}_t{temp}_sample.png')
    plot_grand_z(mask_zs, Attributes().headers, save_dir + f'/k{kept}_t{temp}_mask.png')
    del x, z, mask_zs

def plot_grand_z(grand_zs, names, filename, n_rows_cols=(6, 9), norm='img'):
    # mpl.rc('text', usetex=True)
    # mpl.rcParams['text.latex.preamble']=[r"\boldmath"]

    n_rows, n_cols = n_rows_cols
    fig, axs = plt.subplots(n_rows, n_cols, sharex='all', sharey='all', figsize=(16, 10))

    n = 0
    for col in range(n_cols):
        for row in range(n_rows):
            if n == grand_zs.shape[0]:
                break

            img_z = np.moveaxis(grand_zs[n].copy(), 0, -1)
            if norm == 'img':
                img_z = (img_z - img_z.min()) / (img_z.max() - img_z.min())
            elif norm == 'array':
                img_z = (img_z - img_z.min()) / (grand_zs.max() - grand_zs.min())
            elif norm == 'arrayb':
                img_z = (img_z - grand_zs.min()) / (grand_zs.max() - grand_zs.min())
            axs[row, col].imshow(img_z)
            ttl = r"Grand-${{z}}$ for {}".format(names[n])
            # axs[row, col].title.set_text(ttl, fontsize='xx-small')
            axs[row, col].set_title(ttl, fontsize='xx-small')
            n += 1
    
    fig.suptitle(r"Grand ${{z}}$ for each celebA attribute.")
    fig.tight_layout()
    fig.subplots_adjust(top=.88)
    plt.savefig(filename, bbox_inches='tight')
    print('\nPlot saved to ' + filename)
    plt.close()
    del grand_zs


def plot_grand_z_rgb(grand_zs, names, filename, n_rows_cols=(10, 12)):
    # mpl.rc('text', usetex=True)
    # mpl.rcParams['text.latex.preamble']=[r"\boldmath"]

    n_rows, n_cols = n_rows_cols
    fig, axs = plt.subplots(n_rows, n_cols, sharex='all', sharey='all', figsize=(20, 10))
    # n_cols /= 3 # keep track of channels in for loop.

    # grand_zs = (grand_zs - grand_zs.min()) / (grand_zs.max() - grand_zs.min())

    n = 0
    # import ipdb; ipdb.set_trace()
    for row in range(n_rows):
        for col in range(0, n_cols, 3):
            # import ipdb; ipdb.set_trace()
            z_img = grand_zs[n].copy() 
            for n_ch in range(3): 
                z_base_img = np.zeros(shape=(64, 64, 3), dtype=np.float64)
                ch_img = z_img[n_ch].copy() 
                ch_img = (ch_img - ch_img.min()) / (ch_img.max() - ch_img.min())
                z_base_img[:,:,n_ch] = ch_img
                axs[row, col + n_ch].imshow(z_base_img) # np.moveaxis(z_img, 0, -1) )
                if n_ch == 1:
                    ttl = r"Grand-${{z}}$ for {}".format(names[n])
                    # axs[row, col].title.set_text(ttl, fontsize='xx-small')
                    axs[row, col + n_ch].set_title(ttl, fontsize='xx-small')
            n += 1
    
    fig.suptitle(r"Grand ${{z}}$ for each celebA attribute.")
    fig.tight_layout()
    fig.subplots_adjust(top=.88)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print('\nPlot saved to ' + filename)
    del grand_zs

def y_distance_z(stats, n_categories=10):
    ''' take stats file, returns a similarity matrix
        for distances computed item-wise (pixel-wise).'''
    n_att = len(att.columns)
    distances = np.zeros(shape=(n_att, n_att))
    if joint and measure:
        raise ValueError("set either joint or measure argument, not both.")
    
    # z, y = label_zs(stats['z'])
    n_measures = 2
    out_size = [n_measures] + [stats['z'][0].shape[1], n_categories, n_categories]
    print(f'computing pixel-wise distances. Output matrix sized: {out_size}')
    pixelwise_mean = [np.mean(z, axis=0) for z in stats['z']]
    pixelwise_std = [np.std(z, axis=0) for z in stats['z']]

    y_distance_mean_std = np.zeros(out_size)
    
    for k, measure in enumerate([pixelwise_mean, pixelwise_std]):
        for d_i, m_i in enumerate(measure):
            for d_j, m_j in enumerate(measure):
                distance = np.linalg.norm(m_i - m_j)
                y_distance_mean_std[k, :, d_i, d_j] = distance
    
    return y_distance_mean_std


def calculate_distance(stats, att, joint=False, measure=None):
    n_att = len(att.columns)
    distances = np.zeros(shape=(n_att, n_att))
    if joint and measure:
        raise ValueError("set either joint or measure argument, not both.")


    if joint:
        joint_stats = []
        # stds, means = stats['std'], stats['mean'] # 2 lists
        std_mean = np.concatenate([stats['std'][:,np.newaxis],
                                     stats['mean'][:,np.newaxis]], axis=1)
        # check if size == 19962 * 2
        for i in range(n_att):
            mask_i = att.iloc[:, i].astype(bool)
            s_m_i = std_mean[mask_i, :]

            for j in range(n_att):
                mask_j = att.iloc[:, j].astype(bool)
                s_m_j = std_mean[mask_j, :]

                min_rows = np.min((s_m_i.shape[0], s_m_j.shape[0]))
                distance = np.linalg.norm(s_m_i[:min_rows] - s_m_j[:min_rows])
                distances[i, j] = distance
    else:
        for i in range(n_att):
            mask_i = att.iloc[:, i].astype(bool)
            meas_i = stats[measure][mask_i]
            for j in range(n_att):
                mask_j = att.iloc[:, j].astype(bool)
                meas_j = stats[measure][mask_j]
                min_rows = np.min((meas_i.shape[0], meas_j.shape[0]))
                distance = np.linalg.norm(meas_i[:min_rows] - meas_j[:min_rows])
                distances[i, j] = distance
    return distances

def heatmap(square_mtx, att, filename, plot_title="Magnitude of distance between digits"):

    n_att = len(att.columns)

    fig, ax = plt.subplots(figsize=(10, 10))
    # configure main plot
    # show img, removing 0's to center color palette distribution. 
    norm_sq_mtx = square_mtx.copy()

    if square_mtx.shape[0] == 1: 
        # for new y_distance_z function
        square_mtx = square_mtx.reshape(square_mtx.shape[1:])
        norm_sq_mtx = norm_sq_mtx.reshape(norm_sq_mtx.shape[1:])

    # remove diagonal
    diagonal_sel = np.diag_indices(n_att)
    norm_sq_mtx[diagonal_sel] = None
    # subtract lowest value and divide
    norm_sq_mtx -= np.nanmin(norm_sq_mtx)
    norm_sq_mtx /= np.nanmax(norm_sq_mtx)
    im = ax.imshow(norm_sq_mtx, cmap="plasma")

    ax.set_xticks(range(n_att))
    ax.set_xticklabels(att.columns, rotation='vertical', fontsize='xx-small')
    ax.set_yticks(range(n_att))
    ax.set_yticklabels(att.columns, fontsize='xx-small')
    # ax.xticks(range(n_att), 

    # annotate values within squares
    for i in range(n_att):
        for j in range(n_att):
            if i != j:
                val = norm_sq_mtx[i, j]
                col = 'w' if val < 0.6 else 'b'
                text = ax.text(j, i, "{:.2f}".format(square_mtx[i, j]),
                       ha='center', va='center', color=col, size='xx-small')
            else:
                break
    
    ax.set_title(plot_title)
    fig.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f'Plot saved to {filename}')




def violin_eachdigit(stats, att, filename, n_epoch):
    positions = [i for i in range(len(att.columns))]
    stds, means = [stats['std'], stats['mean']]

    fig = plt.figure()
    ax = fig.add_subplot()
    plt.title('z space stats for model at epoch {}'.format(n_epoch))
    plt.ylabel('Average $z$ values')
    plt.xlabel('Digits')
    
    # axs.set_title('Stats for epoch {} overview'.format(n_epoch))
    # 1. average std per digit
    std_each_digit = [np.mean(std) for std in stds]
    # 2. vector means for each digit
    mean_each_digit = [np.mean(m) for m in means]
    strings = ax.violinplot(dataset=means, positions=positions, showmedians=True, showextrema=False)

    for i, b in enumerate(strings['bodies']):
        b.set_facecolor(colors[i])
        b.set_edgecolor(color_variant(colors[i]))
        b.set_alpha(.6)

    '''
    axs[row,col].annotate('$\mu$: {:.2f}\n$\sigma^2$: {:.2f}'.format(ctrd_x, ctrd_y), (ctrd_x, ctrd_y))
    '''
    # fig.suptitle('Stats for epoch {} overview'.format(n_epoch))
    fig.tight_layout()
    # fig.subplots_adjust(top=.70) # .88
    #for dig in range(10):
    #	plt.scatter(means[dig], stds[dig], c=colors[dig], label=str(dig), alpha=.8, s=.8)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print('\nPlot saved to ' + filename)

class Attributes:
    ''' minimal accessory class for correct 
    import/export of goods and dataframes with
    proper column labeling for celeba attributes'''

    def __init__(self):
        self.filename='data/1_den_celeba/attr_y.csv'
        self.headers = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]
        self.colors = None # for now XXX

    def fetch(self, filename='data/1_den_celeba/attr_y.pkl'):
        # make it pandas dataframe
        if filename.endswith('.pkl') or filename.endswith('.pickle'):
            self.df = pd.read_pickle(filename)
        elif filename.endswith('.csv'):
            self.df_csv = pd.read_csv(filename, index_col=False, dtype=np.int8)
        print('Fetched dataframe from file {}.'.format(filename))
        return self.df

    def make_and_save_dataframe(self, y, filename='data/1_den_celeba/attr_y.pkl'):
        df = pd.DataFrame(y, columns=self.headers, dtype=np.int8)
        if filename.endswith('.pkl'):
            df.to_pickle(filename) # this should be it. Nice and simple.
        elif filename.endswith('.csv'):
            df.to_csv(filename, index=False)
        print('saved celeba attributes to {}'.format(filename))
        return df

    # def attribute_in_z(self, a):

    # 	if isinstance(a, int):
    # 		if a >= 0 and a <= 40:
    # 			attribute = self.headers[a]
    # 		else: raise ValueError
    # 	elif isinstance(a, str):
    # 		if a not in self.headers:
    # 			raise ValueError
    # 	else:
    # 		raise ValueError('int > 0 and < than 0, or one of the following strings: {}. Got {} instead.'.format(self.headers, a)



def tellme_ys(net, loader, device):

    with torch.no_grad():
        with tqdm(total=len(loader.dataset)) as progress:
            attr_y = np.array([]).reshape(0, 40) # for i in range(10)

            for _, y in loader:
                attr_y = np.concatenate(( attr_y, y.to('cpu').detach().numpy() ))
                progress.update(y.size(0))

    return attr_y

def scatter_attr(stats, att, filename, n_epoch):
    stds, means = [stats['std'], stats['mean']]
    # Steps to replicate for each subplot:
    # plt.title('z space stats for model at epoch {}'.format(n_epoch))
    # plt.xlabel('mean')
    # plt.ylabel('std')
    nrows, ncols = (5, 8) # = 40
    fig, axs = plt.subplots(nrows, ncols, sharex='all', sharey='all', figsize=(10, 7))
    
    cmap = plt.get_cmap('gist_rainbow')
    # cmaplist = [cmap(i) for i in range(cmap.N)]
    # cmap = LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = np.linspace(0, 39, 40)
    normidx = BoundaryNorm(bounds, cmap.N)
    
    # for n_att in range(len(att.columns)):
    n_att = 0.
    for col in range(ncols):
        for row in range(nrows):
            att_name = att.columns[row+col*nrows]
            mus = means[att[att_name].astype(bool)]
            att_ds_n = mus.shape[0]
            sigmas = stds[att[att_name].astype(bool)]
            # print('{}: {}, {}'.format(att_name, mus.shape, sigmas.shape))
            axs[row,col].set_title(att_name+ f' $n={att_ds_n}$' , fontsize='xx-small')
            axs[row,col].scatter(mus, sigmas, s=1,
                                 cmap=cmap, c=np.repeat(n_att, mus.shape[0]),
                                 norm=normidx, label=att_name)
            # ctrd_x = np.mean(means[dig])
            # ctrd_y = np.mean(stds[dig])
            # axs[row,col].scatter(ctrd_x, ctrd_y, c=color_variant(colors[dig]))
            if row == nrows-1:
                axs[row,col].set_xlabel('means', fontsize='small')
            if col == 0:
                axs[row,col].set_ylabel('std', fontsize='small')
            # axs[row,col].annotate('$\mu$: {:.2f}\n$\sigma$: {:.2f}'.format(
            # 	                           ctrd_x, ctrd_y), (ctrd_x, ctrd_y))
            n_att += 1 


    fig.suptitle('Stats for epoch {} overview'.format(n_epoch))
    fig.tight_layout()
    fig.subplots_adjust(top=.88)
    plt.savefig(filename)
    plt.close()
    print('\nPlot saved to ' + filename)


def color_variant(hex_color, brightness_offset=-50):
    """ takes a color like #87c95f and produces a lighter or darker variant """
    if len(hex_color) != 7:
        raise Exception("Passed %s into color_variant(), needs to be in #87c95f format." % hex_color)
    rgb_hex = [hex_color[x:x+2] for x in [1, 3, 5]]
    new_rgb_int = [int(hex_value, 16) + brightness_offset for hex_value in rgb_hex]
    new_rgb_int = [min([255, max([0, i])]) for i in new_rgb_int] # make sure new values are between 0 and 255
    # hex() produces "0x88", we want just "88"
    hex_string = [hex(i)[2:] for i in new_rgb_int]
    hex_string = [val if len(val) == 2 else "0" + val for val in hex_string]
    return "#" + "".join(hex_string)


def scatter_all(stats, filename, n_epoch):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
    fig, ax = plt.subplots()
    stds, means = [stats['std'], stats['mean']]
    plt.title('z space stats for model at epoch {}'.format(n_epoch))
    plt.xlabel('mean')
    plt.ylabel('std')

    ax.scatter(means, stds, c='#fab111', alpha=.95, s=.8)
    # ax.legend()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print('\nPlot saved to: ' + filename)


def grand_z(stats, att, filename=None):

    zspace = stats['z']

    n_att = len(att.columns)
    all_zs = np.zeros(shape=(n_att, 3, 64, 64))

    for a in range(n_att):
        a_z = zspace[att.iloc[:, a].astype(bool)]
        grand_nz = np.mean(a_z, axis=0)
        all_zs[a] = grand_nz
    
    return all_zs


def track_z_celeba(net, loader, device, loss_fn, track_x=False, n_bits=5,
                     y_all_p = 'data/1_den_celeba/attr_y.pkl', **kwargs):

    net.eval()
    if type(net.module).__name__ == 'Glow':
        n_bins = 2. ** n_bits

    # arrays for concatenation
    stds_z = np.array([]) 
    means_z = np.array([]) 
    axes = [1, 2, 3]
    # per channel measures
    ch_stds_z = np.array([], dtype=np.float32).reshape(0, 3)
    ch_means_z = np.array([], dtype=np.float32).reshape(0, 3)
    ch_axes = [2, 3]
    celeb_z = np.array([], dtype=np.float32).reshape(0, 3, 64, 64)
    if track_x:
        celeb_x = np.array([], dtype=np.float32).reshape(0, 3, 64, 64)

    with tqdm(total=len(loader.dataset)) as progress:
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                if type(net.module).__name__ == 'Glow':
                    log_p, logdet, z = net(x + torch.rand_like(x) / n_bins)
                    z = net(z, reverse=True, reconstruct=True)
                    # logdet = logdet.mean()
                    # loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
                elif type(net.module).__name__ == 'RealNVP':
                    z, sldj = net(x, reverse=False)
                    # loss = loss_fn(z, sldj)
                means = z.mean(axis=axes) # [1, 2, 3]) # `1` is channel.
                stds = z.std(axis=axes) # dim=[1, 2, 3])
                channel_means = z.mean(axis=ch_axes) # [1, 2, 3]) # `1` is channel.
                channel_stds = z.std(axis=ch_axes) # dim=[1, 2, 3])

                means_z = np.concatenate(( means_z, means.to('cpu').detach().numpy() ))
                stds_z = np.concatenate(( stds_z, stds.to('cpu').detach().numpy() ))

                ch_means_z = np.concatenate(( ch_means_z, channel_means.to('cpu').detach().numpy() ))
                ch_stds_z = np.concatenate(( ch_stds_z, channel_stds.to('cpu').detach().numpy() ))
                # concatenate whole z space. 
                celeb_z = np.concatenate(( celeb_z, z.to('cpu').detach().numpy() ))
                if track_x:
                    celeb_x = np.concatenate(( celeb_x, x.to('cpu').detach().numpy() ))


                # progress.set_postfix(loss=loss_meter.avg,
                # 							 bpd=bpd_meter.avg)
                progress.update(x.size(0))
    # add loss.avg() and bpd.avg() somewhere in the plot.
    
    if track_x:
        return {'std': stds_z, 'mean': means_z, 'z': celeb_z, 
                    'ch_std': ch_stds_z, 'ch_mean': ch_means_z}, {'x': celeb_x}
    else:
        return {'std': stds_z, 'mean': means_z, 'z': celeb_z, 
                    'ch_std': ch_stds_z, 'ch_mean': ch_means_z}


def train(epoch, net, trainloader, device, optimizer, loss_fn, max_grad_norm):
    print('\nEpoch: %d' % epoch)
    net.train()
    loss_meter = util.AverageMeter()
    bpd_meter = util.AverageMeter()
    with tqdm(total=len(trainloader.dataset)) as progress_bar:
        for x, _ in trainloader:
            x = x.to(device)
            optimizer.zero_grad()
            z, sldj = net(x, reverse=False)
            loss = loss_fn(z, sldj)
            loss_meter.update(loss.item(), x.size(0))
            loss.backward()
            util.clip_grad_norm(optimizer, max_grad_norm)
            optimizer.step()
            bpd_meter.update(util.bits_per_dim(x, loss_meter.avg))

            progress_bar.set_postfix(loss=loss_meter.avg,
                                                             bpd=bpd_meter.avg)
            progress_bar.update(x.size(0))
            #debugopt:
    return {'train_loss': loss_meter.avg, 
             'train_bpd': bpd_meter.avg}


    # plot x and z
    images_concat = torchvision.utils.make_grid(images, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
    z_concat = torchvision.utils.make_grid(latent_z, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
    torchvision.utils.save_image(images_concat, save_dir+'/x.png')
    torchvision.utils.save_image(z_concat, save_dir+'/z.png')

    # with open(save_dir+'/z.pkl', 'wb') as z_serialize:
    # 	pickle.dump(latent_z, z_serialize)
    torch.save(latent_z, f = save_dir+'/z.pkl')


    # dict keys as returned by "train"
    train_loss = args['train_loss']
    train_bpd = args['train_bpd']
    report = [epoch, loss_meter.avg, bpd_meter.avg] + [train_loss, train_bpd]

    with open('{}/log'.format(dir_samples), 'a') as l:
        report = ", ".join([str(m) for m in report])
        report += "\n"
        print("\nWriting to disk:\n" + report + "At {}".format(dir_samples))
        l.write(report)

def verify_version(fp, version_string):
    ''' helper function to define what version of analysis was performed. 
    Used to select a model without given analysis version #.'''
    if os.path.isfile(fp):
        with open(fp, 'r+') as f:
            l = f.readline()

            while l:
                print("debugging value for l: " + l)

                if l.strip('\n') == version_string:
                    # if analysis # `version_string` was completed before:
                    print("Matched version: " + version_string)
                    return True
                    break
                else:
                    # TODO: fix eternal loop because f doesn't validate the while loop.
                    instead = l.strip()
                    print(f'Unmatched: {instead} != {version_string} ({fp}).', end='')
                    print(' Continuing...')
                l = f.readline()
        return False
    else: 
        print(f'File {fp} not found.')
        return False # if file doesn't exist.

def select_model(model_root, analyse_version, vmarker_fn='/version', 
                     epoch_range=(430, 690), test=False, granularity=10, figures=6):
    ''' Select EPOCH according to version specifications. (change function name)
    Out: 
        - fp_vmarker : int --  file containing `version`
        - fp_model : str -- file to model.pth.tar
        '''
    verified_ver = True
    while verified_ver:
        if test:
            assert isinstance(test, int), 'Epoch must be int'
            model_epoch = test
        else:
            model_epoch = randrange(epoch_range[0], epoch_range[1], granularity) # TODO only produce n%10==0
        fp_model_root = model_root + f'/epoch_{str(model_epoch).zfill(figures)}' #  + str(model_epoch)
        fp_vmarker = fp_model_root + vmarker_fn
        if test:
            break
        else:
            verified_ver = verify_version(fp_vmarker, str(analyse_version))
    fp_model = fp_model_root + '/model.pth.tar'
    print('selected model at {}th epoch'.format(model_epoch))
    return fp_model_root, fp_model, fp_vmarker

def mark_version(version_str, fp_vmarker, finish=False, sep='-'):
    ''' write first and last parts of `version_str` to `fp_vmarker`
    Args:
        version_str
        fp_vmarker: str -- version marker file path
        finish: final call, else False
        sep: define first and last parts of `version_str`.
    '''
    vmarker = version_str.split(sep)
    m = open(fp_vmarker, 'a')
    if not finish:
        vmarker[0] += sep
        m.write('\n'+vmarker[0])
    else:
        m.write(vmarker[1]+'\n')
        # must end with a newline. byebye!
    m.close()

# def cleanup_version_f(fp_vmarker):
# 
# 	tmp_fp_vmarker = '/home/mao540/tmp_realnvp' + fp_vmarker.replace('/', '')
# 
# 	with open(fp_vmarker, 'r') as v:
# 		with open(tmp_fp_vmarker, 'w') as t:
# 			for l in v:
# 				stripped = l.strip()
# 				if stripped == 'V-' or stripped == '':
# 					continue
# 				else:
# 					t.write(l)
# 	shutil.move(tmp_fp_vmarker, fp_vmarker)
    


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

def load_network(model_dir, device, args):
    if args.net == 'glow':
        from model import Glow
        net = Glow(3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu)
        from train_like import calc_loss
        loss_fn = calc_loss
    elif args.net in ['densenet', 'resnet']:
        net = RealNVP( **filter_args(args.__dict__) )
        from models import RealNVP, RealNVPLoss
        loss_fn = RealNVPLoss()

    net = net.to(device)
    if str(device).startswith('cuda'):
        net = torch.nn.DataParallel(net, args.gpu_ids)
        cudnn.benchmark = args.benchmark

    # load checkpoint
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


def filter_args(arg_dict, desired_fields=None):
    """only pass to network architecture relevant fields."""
    if not desired_fields:
        desired_fields = ['net_type', 'num_scales', 'in_channels', 'mid_channels', 'num_levels']
    return {k:arg_dict[k] for k in desired_fields if k in arg_dict}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RealNVP on CIFAR-10')

    # parser.add_argument('--num_samples', default=121, type=int, help='Number of samples at test time')


    # Analysis
    version_ = 'V-1.1' # only change from 1.0: copy of rZ[idx] in plot_rZ
    # General architecture parameters
    net_ = 'glow'
    dataset_ = 'celeba'
    gpus_ = '[0]'
    force_ = False
    # only needed once for all epochs. 
    ys_ = False
    track_x_ = False


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

    parser.add_argument('--track_x', action='store_true', default=track_x_)
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
    parser.add_argument('--gpu_ids', default=gpus_, type=eval, help='IDs of GPUs to use')
    parser.add_argument('--benchmark', action='store_true', help='Turn on CUDNN benchmarking')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader threads')

    # model_meta_stuff = select_model(root_dir_, version_, test=700)
    main(parser.parse_args())

