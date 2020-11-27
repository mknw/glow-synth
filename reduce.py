#!/var/scratch/mao540/miniconda3/envs/maip-venv/bin/python
# maybe

import torch 

# from matplotlib.colors import LinearSegmentedColormap
# from matplotlib.colors import BoundaryNorm as BoundaryNorm

# import util
import os
# from mnist_simil import cleanup_version_f
from config.config import ConfWrap
from synthesyzer  import Synthesizer

from utils import ArchError
from load_data import load_network, Attributes, select_model
import pickle
from render import save_dataset_reduction
import numpy as np

def main(C, epochs=[160000], save=True):

    p_value_counts = dict()
    for e in epochs:
        model_meta_stuff = select_model(C.training.root_dir, test_epoch=e)
        # model_root_fp, model_fp, vmarker_fp = model_meta_stuff
        # mark_version(C.version, vmarker_fp) # echo '\nV-' >> vmarker_fp

        # if not os.path.isfile(pcount_fn) or force or C.kde:
        analyse_epoch(C, model_meta_stuff)
        # mark_version(C.version, vmarker_fp, finish=True) # echo '0.2' >> vmarker_fp


def analyse_epoch(C, model_meta_stuff = None):
    device = torch.device("cuda:0" if torch.cuda.is_available() and len(C.net.gpus) > 0 else "cpu")
    print("evaluating on: %s" % device)
    
    model_root_fp, model_fp, vmarker_fp = model_meta_stuff

    epoch = model_root_fp.split('_')[-1]
    # Load statistics:
    stats_filename = model_root_fp + '/z_mean_std.pkl'
    stats = torch.load(stats_filename)

    ''' filepaths '''
    compr_fp = model_root_fp + '/compr'

    lstify = lambda s: [s] if isinstance(s, str) else s
    maketree = lambda l: [os.makedirs(p, exist_ok=True) for p in lstify(l)]
    maketree(compr_fp)
    
    if not cached:

        if 'net' in C.steps:
            if 'net' not in dir():
                net, _ = load_network( model_fp, device, C.net)
                net.eval()

        print(f'Loading cached `{C.data}`, skipping computations of mean and std... ',
                end='')
        dataset = dict()
        if 'z' in C.data:
            dataset['z'] = torch.load(f'{model_root_fp}/z_mean_std.pkl')['z']
            data = dataset['z'].reshape(dataset['z'].shape[0], -1)
        if 'x' in C.data:
            dataset['x'] = torch.load(f'{C.training.root_dir}/x.pkl')['x']
            data = dataset['x'].reshape(dataset['x'].shape[0], -1)
        print(f'`{C.data}` stats loaded.')

        attributes = Attributes().fetch()
        # import ipdb; ipdb.set_trace()
        
        reducer = Synthesizer(C, steps = C.steps)
                              # net=net, device=device)
        reducer.fit(data)
        ''' x -> low_dim '''
        # TODO: IN- and OUT- of set reduction / transformation.

        # red_data = reducer.transform(data, show_steps='all')
        # import ipdb; ipdb.set_trace()

        var_exp_ratio = reducer.models['pca'].explained_variance_ratio_
        if save_cache:
            with open(cache_fn, 'wb') as f:
                pickle.dump(attributes, f)
                pickle.dump(red_data, f)
                pickle.dump(var_exp_ratio, f)

    else:
        # with open(cache_fn, 'rb') as f:
            # attributes = pickle.load(f)
            # red_data = pickle.load(f)
            # var_exp_ratio = pickle.load(f)
        pass

    # Should we select the number of component as second dimension of 
    # reduced data here or inside the visualization function?
    # if inside, add k argument to save_dataset_reduction()
    # FOR NOW is inside plot_pca in render.py. Not too important anyway

    data_red = reducer.transform(data)
    for i in range(0, 39, 5):

        fns = [f'data/test/{i}_{models}_{version}_4.png' for version in ['legacy', 'new']]

        # save_dataset_reduction(data_red, var_exp_ratio, attributes, k=10, att_ind=i, filename=fns[1])
        save_dataset_reconstruction(data, data_red, model, attributes)

    

    # rec_data = reducer.inverse_transform(red_data[-1], show_steps=-1)




def plot_reduced_dataset(pca, z_s, att, k, att_ind, filename):
    # sort from highest variance
    from sklearn.decomposition import PCA
    if isinstance(pca, PCA):
        components = pca.components_[:k][::-1]
        var_exp = pca.explained_variance_[:k][::-1]
        ratio_var_exp = pca.explained_variance_ratio_[:k][::-1]
        '''z_s, y = label_zs(z_s)'''
        from celeb_simil import subset_attributes, category_from_onehot
        
        sel_att_df = subset_attributes(att, att_ind, overall_indexer=True, complementary=True)
        red_z = pca.transform(z_s[ sel_att_df.iloc[:, -1]].reshape(sel_att_df.shape[0], -1))
        reduced_z = red_z[:, :k][:,::-1]   # PCs['X'].T becomes (306,10000)
    else: # should be type: sklearn.decomposition.PCA
        raise NotImplementedError
    
    from matplotlib import pyplot as plt

    symbols = "." # can be used for orthogonal attributes.
    n_pcs = k # can use this to index components and create grid.
    fs = int(n_pcs * 2)
    fig, axs = plt.subplots(n_pcs, n_pcs, figsize=(fs, fs), sharex='col', sharey='row')
    cmap = plt.cm.winter # get_cmap('Set1')

    # Use subset dataframe turn 1 hot vectors into indices,
    # then add column for "both" categories if overlapping.
    # import ipdb; ipdb.set_trace()
    color_series, overlapping_attributes = category_from_onehot(sel_att_df)
    # color_series += 2 # make it red
    
    for row in range(n_pcs):
        # random permutation of reduced datapoints for 
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
    print(f'Saved to {filename}.')


if __name__ == '__main__':
    C = ConfWrap(fn='config/config.yml')
    # here only for compatibility:
    C.data = ['z']
    # C.version = 'V-C.0'
    C.steps = ['pca']
    C.att_ind = 10
    cached = False
    save_cache = False

    models = '-'.join(C.steps)
    pca_params = '-'.join([str(i) for j in C.pca.items() for i in j])
    data_steps = '-'.join(C.data)

    models = '_'.join([models, pca_params, data_steps])
    # fns = [f'data/test/{models}_{version}_4.png' for version in ['legacy', 'new']]
    # cache_fn = f'data/hybrid_full.pkl'

    main(C)
