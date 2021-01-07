#!/bin/env /var/scratch/mao540/miniconda3/envs/maip-venv/bin/python

#/var/scratch/mao540/miniconda3/envs/maip-venv/bin/python
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
from render import save_dataset_reduction, plot_compression_flow
import numpy as np

def main(C, epochs=[160000], save=True):

    p_value_counts = dict()
    for e in epochs:
        model_meta_stuff = select_model(C.training.root_dir, test_epoch=e)
        # model_root_fp, model_fp, vmarker_fp = model_meta_stuff
        # mark_version(C.version, vmarker_fp) # echo '\nV-' >> vmarker_fp

        # if not os.path.isfile(pcount_fn) or force or C.kde:
        # import ipdb; ipdb.set_trace()
        for kept_out in [True, False]:
            for steps in [['net', 'pca'],['net','pca','umap']]:
                for pca_npcs in [int(np.exp(i)) for i in range(3,10)]:
                    for umap_npcs in [10, 20, 50]:
                        C.pca.n_pcs = pca_npcs
                        C.umap.n_components = umap_npcs
                        C.kept_out = kept_out
                        C.steps = steps
                        analyse_epoch(C, model_meta_stuff)
                        print(f'failed to compute pca: {pca_npcs}, umap: {umap_npcs}')
                        pass
        # mark_version(C.version, vmarker_fp, finish=True) # echo '0.2' >> vmarker_fp


def analyse_epoch(C, model_meta_stuff = None):
    device = torch.device("cuda:0" if torch.cuda.is_available() and len(C.net.gpus) > 0 else "cpu")
    # print("Device: %s" % device)
    
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

    
    
    if not use_cache:

        if 'net' in C.steps and \
            'net' not in dir():
            net, _ = load_network( model_fp, device, C.net)
            net.eval()

        print(f'Loading use_cache `{C.data}`, skipping computations of mean and std... ',
                end='')

        # import ipdb; ipdb.set_trace()
        
        reducer = Synthesizer(C, steps = C.steps,
                              net=net, device=device)
        if save_cache:
            # with open(cache_fn, 'wb') as f:
            # 	pickle.dump(attributes, f)
            # pickle.dump(red_data, f)
            # pickle.dump(var_exp_ratio, f)
            pass

        dataset = dict()
        if 'z' in C.data:
            dataset['z'] = torch.load(f'{model_root_fp}/z_mean_std.pkl')['z']
            data_z = dataset['z'].reshape(dataset['z'].shape[0], -1)
        if 'x' in C.data:
            dataset['x'] = torch.load(f'{C.training.root_dir}/x.pkl')['x']
            data_x = dataset['x'].reshape(dataset['x'].shape[0], -1)
        print(f'`{C.data}` stats loaded.')
    attributes = Attributes().fetch()

    # import ipdb; ipdb.set_trace()
    basename = make_basename(C)
    if use_cache:
        save_reconstr_v(attributes,use_cache=use_cache,save_cache=save_cache,
                basename=basename)
    else:
        save_reconstr_v(attributes, data_z, data_x, reducer,
                        use_cache=use_cache, save_cache=save_cache, basename=basename)

def make_basename(C, withtime=False):
    time = ''
    if withtime: 
        import datetime as dt
        time += str(dt.datetime.now()).split(sep='.')[0].replace('2021-', '').replace(' ', '_')
    basename = f'{C.training.root_dir}/reupsam'
    basename += f'/{time}syn'
    if 'umap' in C.steps:
        basename += f'_uc{C.umap.n_components}'
    if 'pca' in C.steps:
        basename += f'_pc{C.pca.n_pcs}'
    return basename


def pca_reduction(reducer, data):
    # TODO 1 move reducer.fit() method outside of function
    # TODO 2 test for cases where len(att_ind) > 1.
    reducer.fit(data)
    ''' x -> low_dim '''
    # TODO: IN- and OUT- of set reduction / transformation.
    var_exp_ratio = reducer.models['pca'].explained_variance_ratio_
    data_red = reducer.transform(data)
    for i in range(0, 39, 5):
        fns = [f'data/test/{i}_{models}_{version}_4.png' for version in ['legacy', 'new']]
        save_dataset_reduction(data_red, var_exp_ratio, attributes, k=10, att_ind=i, filename=fns[1])

def save_reconstr_v(attributes, data=None, data_x=None, reducer=None,
                      use_cache=False, save_cache=False, basename=None, ko=True,
                            resample=True):
    
    ''' Save x's and z's, visualized pre (original) 
    and post (reconstructed) dimensionality reduction.'''

    import random 
    att_ind = random.randint(0, 39)
    ko = 'ko' if ko else 'no-ko'
    filename = f'{basename}_{att_ind}_{ko}.png'
    cache_fn = f'{basename}_{att_ind}_{ko}.pkl'

    # select attributes
    if not use_cache:
        # att_ind = list(range(0, 40, 10))
        kept_out_df, kept_out_idcs = attributes.pick_last_n_per_attribute(att_ind, n=16)

        # split dataset
        z_s = data[kept_out_idcs].copy()
        x_s = data_x[kept_out_idcs].copy()

        if ko:
            training_data = np.delete(data, kept_out_idcs, axis=0)
        else:
            training_data = data.copy()

        del data_x; del data
        try:
            # fit PCA (and UMAP)
            reducer.fit(training_data)
        except:
            import ipdb; ipdb.set_trace()
        del training_data


        # TODO: replace show_steps arguments with argument selection
        red_data = reducer.transform(z_s, show_steps='all')

        # the last element of red(uced)_data is the lower level representation.
        rec_data = reducer.inverse_transform(red_data[-1], show_steps='all',
                                          resample=resample)

        # step_vector is equal to: [x_s, z_s, PCs, UMAP_embeddings, rec_z, rec_x]
        step_vector = [x_s, z_s] + [reducer.models['pca'].components_] + rec_data
        if 'umap' not in C.steps:
            # placeholder vector.
            step_vector = [x_s, z_s] + [reducer.models['pca'].components_] \
                          + [reducer.models['pca'].components_] + rec_data
        
        if save_cache:
            with open(cache_fn, 'wb') as f:
                pickle.dump(step_vector, f)
    else:
        with open(cache_fn, 'rb') as f:
            step_vector = pickle.load(f)

    plot_compression_flow(step_vector, filename)
    
    print('done.')
    

def plot_reduced_dataset(pca, z_s, att, k, att_ind, filename):
    import warnings
    raise warnings.DeprecationWarning
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
    C = ConfWrap(fn='config/r_config.yml')
    # here only for compatibility:
    C.data = ['z' , 'x']
    # C.version = 'V-C.0'
    # C.steps = ['net', 'pca'] # , 'umap']
    # C.att_ind = 10
    use_cache = False
    save_cache = True

    # models = '-'.join(C.steps)
    # pca_params = '-'.join([str(i) for j in C.pca.items() for i in j])
    # data_steps = '-'.join(C.data)

    # models = '_'.join([models, pca_params, data_steps])
    # fns = [f'data/test/{models}_{version}_4.png' for version in ['legacy', 'new']]

    main(C)
