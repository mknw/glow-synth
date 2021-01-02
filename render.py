#!/var/scratch/mao540/miniconda3/envs/maip-venv/bin/python

from itertools import product
from load_data import Attributes
from matplotlib import pyplot as plt
import numpy as np

'''PCA'''

def save_dataset_reduction(reduced_data, var_exp, attributes, k, att_ind, filename):
    ''' Obtain categories and saves color-coded scatterplots along 
    the first Principal Components of the provided dataset'''

    # select subset of 
    (category_array, category_map) = attributes.categorise_subset(att_ind)
    # z_subset = pca_reduced_z[onehot_subset]

    # indices = np.random.permutation(pca_reduced_z.shape[0])
    # pca_reduced_z = np.take(pca_reduced_z, indices, axis=0)
    # category_array = np.take(category_array, indices)
    plot_pca(reduced_data, var_exp, category_array, category_map, filename, k=k)

def plot_pca(reduced_data, var_exp, category_array, category_map, filename, k=5):
    fs = int(k*2)

    var_exp = var_exp[k-1::-1]
    reduced_data = reduced_data[:, k-1::-1]


    fig, axs = plt.subplots(k, k, figsize=(fs, fs), sharex='col', sharey='row')
    cmap = plt.cm.winter.reversed()


    for row in range(k):
        for col in range(k):
            if row > col:
                path_c = axs[row, col].scatter(
                         reduced_data[:, col], reduced_data[:,row],
                         c=category_array, cmap=cmap, s=.5, alpha=.6)
                VE_annotation = '% VE:\nC{}={:.2f}\nC{}={:.2f}'.format(
                                  k - row, var_exp[row]*100,
                                  k-col, var_exp[col]*100)
                axs[row, col].annotate(VE_annotation, xy=(0.625, 0.625), xycoords='axes fraction', fontsize='xx-small')
                if row == k-1:
                    axs[row, col].set_xlabel(f'component {k-col}') 
                    axs[row, col].tick_params(axis='x', reset=True, labelsize='x-small')
                if col == 0:
                    axs[row, col].set_ylabel(f'component {k-row}')
                    axs[row, col].tick_params(axis='y', reset=True, labelsize='x-small')
            else:
                axs[row, col].remove()
                axs[row, col] = None

    handles, labels = path_c.legend_elements(prop='colors')
    labels = list(category_map.keys())
    plt.legend(handles, labels, bbox_to_anchor=(.625, .625), loc="upper right", 
               bbox_transform=fig.transFigure)

    # use category_map to name categories
    fig.tight_layout()
    fig.subplots_adjust(top=.88)
    plt.show()
    plt.savefig(filename, bbox_inches='tight')
    print(f'visualization saved to path: {filename}')
    plt.close()

def save_dataset_reconstruction(x, reduced_data, attributes, n_samples=5):
    
    n_red = pca_reduced_z.shape[0]
    n_x = x.shape[0]
    n_att = attributes.df.shape[0]

    if n_red != n_x or n_red != n_att:
        raise ValueError

    # select attributes
    att_ind = 5
    (category_array, category_map) = attributes.subset(att_ind)

    # select n examples from attributes, reduced_data, x.

    # according to the number and types of model, plot the correct type
    # of reconstruction.

    # run reconstruction on each selected z to obtain z_reduced



def plot_compression_flow(data_arrays, filename, steps=None):
    '''
    Args: 
        data_arrays containing: [x_s, z_s, PCs, UMAP_embeddings, rec_z, rec_x]
    '''

    from matplotlib.gridspec import GridSpec
    # import ipdb; ipdb.set_trace()
    # 2x3 quadrants. 
    # x'  |<- rec_z  |<- umap_scatter|
    # x ->|  z     ->|  eigenfaces ↑ |
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(14, 9, fig) 

    xs_axs = np.zeros((4, 4), dtype=object)
    eigenf_axs= np.zeros((4, 4), dtype=object)
    pca_axs = np.zeros((4, 4), dtype=object)
    rec_z_axs = np.zeros((4, 4), dtype=object)
    rec_x_axs = np.zeros((4, 4), dtype=object)

    for i in range(4):
        for j in range(4):
            xs_axs[i,j] = fig.add_subplot(gs[i,j])
            eigenf_axs[i,j] = fig.add_subplot(gs[5+i, j])
            pca_axs[i,j] = fig.add_subplot(gs[10+i, j])
            rec_z_axs[i,j]  = fig.add_subplot(gs[5+i, 5+j])
            rec_x_axs[i,j]  = fig.add_subplot(gs[i, 5+j])

    umap_emb_ax = fig.add_subplot(gs[10:, 5:])
    # import ipdb; ipdb.set_trace()
    # reformat images
    axes = [xs_axs, eigenf_axs, pca_axs,  umap_emb_ax, rec_z_axs, rec_x_axs]


    # always show X (last column)
    for i, arr in enumerate(data_arrays):
        # std = (i in [0, 5])
        build_quadrant(arr, axes[i], i) # , std=False)

    # build rectangles by providing gridspec?

    plt.savefig(filename)
    plt.close()
    print(f'plot saved to: {filename}')


def build_quadrant(data, axes, i, std=None):

    if i in [3]: # pca, umap
        try:
            axes.scatter(data[:, 0], data[:, 1])
        except:
            import ipdb; ipdb.set_trace()
        return

    if i not in [3]: # setup image format.
        if i in [1, 4]: # Z + rec_Z
            d_min = data.min(1, keepdims=True)
            d_std = data.std(1, keepdims=True)
            data = (data - d_min) / d_std
            d_max = data.max(1, keepdims=True)
            data /= d_max
            # import ipdb; ipdb.set_trace()
        data = np.moveaxis(data.reshape(-1, 3, 64, 64), 1, -1)
    
    col_count = 0
    for col in range(4):
        for row in range(4):
            img = data[row + 4*col_count].copy()
            axes[col, row].imshow(img, interpolation='none')
            axes[col, row].set(xticks=[], yticks=[])
        col_count += 1



def plot_reconstruction(reduced_z, att, filename, n_examples, net, device,
                         selected_attributes=5):

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

    for col in range(ncols):

        # pick selected attribute N.
        n_att = selected_attributes[col]
        ori_Z = z_s[att.iloc[:, n_att].astype(bool)].astype(np.float32)
        celeb_idx = np.random.randint(ori_Z.shape[0], size=nrows)
        ori_Z = ori_Z[celeb_idx].copy()
        
        ### original Zs
        # keep original Zs for plotting; oriZ for generation
        # variable without underscores `_` are used for the GPU (pytorch).
        # Transform with var. explained by PCs

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
    pass

def nul_fun():
    ''' net '''
    if 'net' in reducer.steps:
        # show 'Z'
        # append to v.
        axs = build_quadrant(xs, axs, col=-1, n=10)
        axs = build_quadrant(reduced_data[0], axs, n=10)

    ''' pca '''
    if 'pca' in reducer.steps:
        # x' <- rec_z <- (cov.M * PCs) <- z <- x
        # show eigenvectors
        pass

    ''' umap '''
    if 'umap' in reducer.steps:
    # x' <- rec_z <- (N-dim. w/ N << p) <- z <- x
        pass

    ''' MEMO: net + pca + umap '''
    # x' <- rec_z <- (cov.M * PCs) <- (N-dim. w/ N << p) <- (PCA_reduction) <- z <- x

    """ generation """
    ''' pca '''
    # x' <- rec_z <- (cov.M * PCs) <- random_z
    ''' umap '''
    # x' <- rec_z <- (N-dim. w/ N << p) <- (random N-dim vector)
    ''' pca + umap '''
    # x' <- rec_z <- (cov.M * PCs) <- (N-dim. w/ N << p) <- 

    n_red = reduced_data.shape[0]
    n_x = x.shape[0]

    n_att = attributes.df.shape[0]
    pass
    
