import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
from MFZ.util import load_stats
from MFZ.dataloader import get_parent_dir

import matplotlib.ticker as ticker

sns.set_theme()

def plot_stats(stats_key, channel = 0 , shuffle = False, save_path = get_parent_dir() / 'Mesh-Float-Zip/figures/final'):
    # plot the stats for a given dataset against each other

    # load the stats whose regex matches the stats_key

    stats_dir = get_parent_dir() / 'Mesh-Float-Zip/stats'

    if channel is None:
        glob_string = f'(?=.*shuffle_{shuffle})(?=.*{stats_key})'
    else:
        glob_string = f'(?=.*shuffle_{shuffle})(?=.*channel_{channel})(?=.*{stats_key})'

        
    pattern = re.compile(glob_string)
    stats_files = []

    for x in (stats_dir / 'mfz').iterdir():
        if pattern.search(x.name):
            stats_files.append(x)

    for x in (stats_dir / 'sz').iterdir():
        if pattern.search(x.name):
            stats_files.append(x)

    for x in (stats_dir / 'zfp').iterdir():
        if pattern.search(x.name):
            stats_files.append(x)

    stats = [load_stats(stats_file) for stats_file in stats_files]

    fig, ax = plt.subplots()

    for i , stat in enumerate(stats):

        stat_name = re.search('mfz|sz|zfp', str(stats_files[i])).group(0)

        ax.plot(stat['comp_ratio'], stat['avg_rel_frob_error'], 'o-' , label=stat_name)


    ax.set_xlabel('Compression Ratio')
    ax.set_ylabel('Relative Frobenius Error')
    #ax.set_xscale('log')
    ax.set_title(f'Channel {channel}: Relative Frobenius Error vs Compression Ratio')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=5))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, numticks=5, subs='all'))
    ax.legend()

    plt.tight_layout()

    plt.savefig(save_path / f'{stats_key}' / f'channel_{channel}_rel_frob_vs_compression_ratio_{stats_key}_shuffle_{shuffle}.png')

    plt.close()


    fig, ax = plt.subplots()

    for i , stat in enumerate(stats):

        stat_name = re.search('mfz|sz|zfp', str(stats_files[i])).group(0)

        ax.plot(stat['comp_ratio'], stat['avg_psnr'], 'o-' ,label=stat_name)

    ax.set_xlabel('Compression Ratio')
    ax.set_ylabel('PSNR')
    ax.set_ylim(0, 120)
    ax.set_xscale('log')
    ax.set_title(f'Channel {channel}: PSNR vs Compression Ratio')
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.legend()

    plt.tight_layout()

    plt.savefig(save_path / f'{stats_key}' / f'channel_{channel}_avg_psnr_vs_compression_ratio_{stats_key}_shuffle_{shuffle}.png')

    plt.close()





def plot_mhb(points, basis, basis_name='test' , save_path = get_parent_dir() / 'Mesh-Float-Zip/figures/final'):

    fig, ax = plt.subplots()

    (save_path / 'mhb' / basis_name).mkdir(parents=True, exist_ok=True)

    for i in range(basis.shape[1]):

        ax.tripcolor(points[:,0], points[:,1], basis[:,i], cmap='vi', shading='gouraud')



        plt.tight_layout()

        plt.savefig(save_path / 'mhb' / basis_name / 'eigenfunction_{i}.png')

        plt.close()