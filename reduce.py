#!/bin/env /var/scratch/mao540/miniconda3/envs/revive/bin/python

#/var/scratch/mao540/miniconda3/envs/maip-venv/bin/python
import torch 
import os
# from mnist_simil import cleanup_version_f
from config.config import ConfWrap
from synthesyzer  import Synthesizer

from utils import ArchError
from load_data import load_network, Attributes, select_model
import pickle
import numpy as np
# Plot analysis:
# from render import save_dataset_reduction, plot_compression_flow

def main(C, epochs=[160000], save=True):

    p_value_counts = dict()
    for e in epochs:
        model_meta_stuff = select_model(C.training.root_dir, test_epoch=e)
        # model_root_fp, model_fp, vmarker_fp = model_meta_stuff
        # mark_version(C.version, vmarker_fp) # echo '\nV-' >> vmarker_fp
        for kept_out in [True, False]:
            C.kept_out = kept_out
            for steps in [['net', 'pca', 'umap'],['net','pca']]:
                C.steps = steps

                for pca_npcs in [int(np.exp(i)) for i in range(3,10)]:
                    C.pca.n_pcs = pca_npcs

                    if 'umap' in steps:
                        umap_npcs_l = [10, 20, 50]
                    else:
                        umap_npcs_l = ['empty']
                    for umap_npcs in umap_npcs_l:
                        C.umap.n_components = umap_npcs
                        analyse_synthesizer(C, model_meta_stuff)
        # mark_version(C.version, vmarker_fp, finish=True) # echo '0.2' >> vmarker_fp


def analyse_synthesizer(C, model_meta_stuff = None):
    import ipdb; ipdb.set_trace()
    print(f'Analysing synth with\
            PCA nps:{C.pca.n_pcs}, UMAP dims: {C.umap.n_components}, ko:{C.kept_out}')

    device = torch.device("cuda:0" if torch.cuda.is_available() and len(C.net.gpus) > 0 else "cpu")
    
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

    if not C.use_cache:
        if 'net' in C.steps and \
            'net' not in dir():
            net, _ = load_network( model_fp, device, C.net)
            net.eval()

        print(f'Loading use_cache `{C.data}`, skipping computations of mean and std... ',
                end='')
        reducer = Synthesizer(C, steps = C.steps,
                              net=net, device=device)
        dataset = dict()
        if 'z' in C.data:
            dataset['z'] = torch.load(f'{model_root_fp}/z_mean_std.pkl')['z']
            data_z = dataset['z'].reshape(dataset['z'].shape[0], -1)
        if 'x' in C.data:
            dataset['x'] = torch.load(f'{C.training.root_dir}/x.pkl')['x']
            data_x = dataset['x'].reshape(dataset['x'].shape[0], -1)
        print(f'`{C.data}` stats loaded.')
    attributes = Attributes().fetch()

    C.basename = make_basename(C)
    if C.use_cache:
        compute_reduction_reupsampling(C, attributes)
    else:
        compute_reduction_reupsampling(C, attributes, data_z, data_x, reducer)

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

def compute_reduction_reupsampling(C, attributes, data=None, data_x=None, reducer=None):
    
    ''' Save x's and z's, visualized pre (original) 
    and post (reconstructed) dimensionality reduction.'''

    # blond vs brown air, smiling vs. wearing hat.
    att_ind = [5, 11, 31, 35]
    ko = '_ko' if C.kept_out else ''
    filename = f'{C.basename}_{att_ind}{ko}.png'
    cache_fn = f'{C.basename}_{att_ind}{ko}.pkl'

    # select attributes
    if not C.use_cache:
        # att_ind = list(range(0, 40, 10))
        kept_out_df, kept_out_idcs = attributes.pick_last_n_per_attribute(att_ind, n=4)
        # split dataset
        z_s = data[kept_out_idcs].copy()
        x_s = data_x[kept_out_idcs].copy()
        att_names = list(kept_out_df.columns)

        if C.kept_out:
            reducer.fit(np.delete(data, kept_out_idcs, axis=0))
        else:
            reducer.fit(data)
        del data_x; del data
        # TODO: replace show_steps arguments with argument selection
        red_data = reducer.transform(z_s, show_steps='all')
        # the last element of red(uced)_data is the lower level representation.
        rec_data = reducer.inverse_transform(red_data[-1], show_steps='all',
                                          resample=C.training.resample)
        if 'umap' in C.steps:
            step_vector = [x_s, z_s] + red_data + rec_data[1:]
        else:
            step_vector = [x_s, z_s] + [reducer.models['pca'].components_] \
                           + red_data + rec_data
        if C.save_cache:
            with open(cache_fn, 'wb') as f:
                pickle.dump(step_vector, f)
    else:
        with open(cache_fn, 'rb') as f:
            step_vector = pickle.load(f)

    # plot_compression_flow(step_vector, filename, att_names, C.steps)
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
    C.use_cache = False
    C.save_cache = True

    # models = '-'.join(C.steps)
    # pca_params = '-'.join([str(i) for j in C.pca.items() for i in j])
    # data_steps = '-'.join(C.data)

    # models = '_'.join([models, pca_params, data_steps])
    # fns = [f'data/test/{models}_{version}_4.png' for version in ['legacy', 'new']]
    main(C)
