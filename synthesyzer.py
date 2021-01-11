#!/var/scratch/mao540/miniconda3/envs/maip-venv/bin/python

import torch 

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

class  Synthesizer(object):
    def __init__(self, C, steps = ['net', 'pca', 'umap'],
                            net=None, device=None,
                            store_config=True):

        np.random.seed(24)
        if isinstance(steps, str):
            steps = [steps]
        elif not isinstance(steps, list):
            raise ValueError

        self.steps = steps
        self.n_steps = len(steps)
        self.models = dict()

        if 'net' in steps:
            assert net is not None, 'net object not provided.'
            assert steps.index('net') == 0, 'Net must be first, if present.'
            self.models['net'] = net
            # self.net_state = 'trained'
            steps = steps[1:]
            if device is not None:
                self.device = device
            else:
                raise ValueError(f'Neural net was given, but device not provided')

        for step in steps:
            if step == 'pca':
                self.models[step] = PCA(n_components=C.pca.n_pcs, whiten = C.pca.whiten,
                                          random_state=42)
            if step == 'umap':
                self.models[step] = UMAP(n_neighbors = C.umap.nn, min_dist = C.umap.min_dist,
                                n_components = C.umap.n_components, random_state=42)
        if store_config:
            self.C = C
            self.channels = C.training.in_channels
            self.img_size = C.training.img_size
            self.batch = C.training.batch_size

    def fit(self, data, skip_l=['net']):
        ''' Main fit method .
        Args: 
            - data: numpy array of size (n x p) '''
        
        for step in self.steps:
            if step in skip_l:
                print(f'{step.capitalize()} already trained. Skipping..')
                continue
            else:
                print(f'fitting {step}')
                if step is not self.steps[-1]:
                    data = self.models[step].fit_transform(data)
                else:
                    self.models[step].fit(data)
        del data
        return

    def transform(self, data, show_steps='all', skip_l=['net']):
        ''' Main transform method.
        Args:
            - data: numpy array of size (n x p), 
                       can be different than the one given to .fit()'''
        # TODO: manage 'levels' for returning reductions at different steps.
        if isinstance(show_steps, int):
            show_steps = [show_steps]
        elif show_steps=='all':
            show_steps = [i for i in range(len(self.steps))]
        elif not isinstance(show_steps, list):
            raise ValueError(f'Cannot show steps of type: {type(show_steps)}')

        selected_steps = [self.steps[step_idx] for step_idx in show_steps]
        out = list()
        
        for step in self.steps:
            if step in skip_l:
                print(f'Skipping {step}...')
                continue
            else:
                print(f'{step.capitalize()} transform...', end='')
                if step == 'net':
                        data = self.net_infer(data)
                else:
                    data = self.models[step].transform(data)

                if step in selected_steps:
                    if len(selected_steps) == 1:
                        return data
                    else:
                        out.append(data)
        return out

    def inverse_transform(self, data, show_steps='all', resample=True, skip_l=[None]):
        
        if isinstance(show_steps, int):
            show_steps = [show_steps]
        elif show_steps=='all':
            show_steps = [i for i in range(len(self.steps))]
        elif not isinstance(show_steps, list):
            raise ValueError(f'Cannot show steps of type: {type(show_steps)}')

        selected_steps = [self.steps[step_idx] for step_idx in show_steps]
        out = list()

        for step in reversed(self.steps):
            if step in skip_l:
                print(f'Skipping {step}...')
                continue
            else:
                print(f'{step.capitalize()} inverse transform...', end='')
                if step == 'net':
                    data = self.net_generate(data, resample=resample)
                else:
                    data = self.models[step].inverse_transform(data)

                if step in selected_steps:
                    out.append(data)
                    if len(selected_steps) == 1:
                        return out[0]
        return out
    
    def net_infer(self, input_data):
        # actual transformation
        # TODO: implement loader  (load_data.load_celeba)
        if input_data.shape[0] <= self.batch:
            batch_size = 1
        else:
            batch_size = self.batch
        sample_size = input_data.shape[0]

        if sample_size % batch_size:
            samples_in_excess = sample_size % batch_size
            print(f'synthesizer.net_ infer() : trimming last {samples_in_excess} datapoints.')
            input_data = input_data[:-samples_in_excess]
        
        
        T = input_data.reshape(
                -1,
                batch_size,
                self.channels,
                self.img_size,
                self.img_size
           )
        
        data_out = np.array([]).reshape(0, input_data.shape[1])

        for batch in T:
            b = torch.from_numpy(batch).to(self.device)
            try:
                log_p, log_det, imgs = self.models['net'](b)
            except RuntimeError as re:
                print(str(re), b.shape)
                raise RuntimeError

            # reconstruct output data
            imgs = self.models['net'](
                                 imgs, reverse=True, reconstruct=True
                                 ) .cpu().detach().numpy(
                                 ).reshape(-1, input_data.shape[1])

            data_out = np.concatenate([data_out, imgs])
        return data_out
    
    def net_generate(self, input_data, resample=True):
        ''' f(z) -> x
        args: input_data (n x p),
        resample: (false if input_data is novel (e.g. randomly created), else true)'''
        # partition input data
        # if data.shape[0] > self.batch:
        # 	data.reshape(-1, self.channels, self.img_size, self.img_size)
        import ipdb; ipdb.set_trace()

        with torch.no_grad():
            data = self.models['net'](
                                      torch.from_numpy(
                                        input_data.reshape(
                                          input_data.shape[0],
                                          self.channels,
                                          self.img_size,
                                          self.img_size
                                          ) # .astype(np.float32)
                                        ).to(self.device),
                                      partition=True
                                     )
            # actual "inverse" transformation
            data = self.models['net'](
                                      data, 
                                      reverse=True,
                                      resample=resample
                                      ).reshape(input_data.shape[0], -1)
        return data.cpu().detach().numpy()

if __name__ == "__main__":
    raise NotImplementedError
