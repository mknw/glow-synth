#!/var/scratch/mao540/miniconda3/envs/maip-venv/bin/python3.7

from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
import pandas


def plot_loss(filename, log_fp='data/res_3-8-32/log', train=False, test=True, hi_epochs=None,
                  ylimits=None, threshold=None, loss=True, bpd=True, names=None):

    if not names:
        names = ['epoch', 'test_loss', 'test_bpd', 'train_loss', 'train_bpd']

    df = pandas.read_csv(log_fp, names=names)

    if threshold:
        df[df>threshold] = None


    fig, ax = plt.subplots(figsize=(10, 7))
    if ylimits:
        plt.ylim(ylimits)


    if train:
        train_alpha = 0.35 if test else 1
        if loss:
            df.plot(kind='line', x='epoch', y='train_loss', color='#cb4b16', alpha=train_alpha, ax=ax) # solarized orange.
        if bpd:
            df.plot(kind='line', x='epoch', y='train_bpd', color='#2aa198', alpha=train_alpha, ax=ax) # solarized cyan.
    if test:
        if loss:
            df.plot(kind='line', x='epoch', y='test_loss', color='#dc322f', linewidth=1, ax=ax) # solarized red.
        if bpd:
            df.plot(kind='line', x='epoch', y='test_bpd', color='#268bd2', linewidth=1, ax=ax) # solarized blue.
    
    if hi_epochs:
        for epoch in hi_epochs:
            # dirty, sexy fix for multiple legends.
            if epoch == hi_epochs[0]:      #'#859900'
                plt.axvline(x=epoch, ymin=0, c='black', ymax=ylimits[1], label='analysed', alpha=0.1) # solarized green.
            else:
                plt.axvline(x=epoch, ymin=0, c='black', ymax=ylimits[1], alpha=0.1) # solarized green.

    plt.tight_layout()

    stats_legend = plt.legend()
    ax = plt.gca().add_artist(stats_legend)
    print(stats_legend)
    
    # analysed_epochs_patch = mpatches.Patch(color='#859900', label='analysed epochs')

    plt.title("Learning stats")
    plt.savefig(filename)


# CelebA params #
glow_celeba_params_loss = {
                                             'log_fp': 'data/1_den_celeba/log',
                                             'filename': './figs/dceleba_loss.png',
                                             'train': True,
                                             'test': True,
                                             'ylimits': (22500, 30000),
                                             'loss': True,
                                             'bpd': False
                                             }

glow_celeba_params_bpd = {
                                            'log_fp': 'data/glow_celeba/log',
                                            'filename': './figs/dceleba_bpd.png',
                                            'train': True,
                                            'test': True,
                                            'ylimits': (2.65, 3.5),
                                            'loss': False,
                                            'bpd': True
                                            }



if __name__ == '__main__':
    # with everything
    glow_celeba_params_loss['names'] = ['loss', 'log p', 'log det', 'LR', 'n imgs']

    plot_loss( **glow_celeba_params_loss )
    plot_loss( **glow_celeba_params_bpd  )





