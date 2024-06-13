from MFZ.tests import sz_compress_time_series
import numpy as np
from MFZ.dataloader import load_neuron_tx, ax1_shuffle, get_parent_dir
from MFZ.util import save_stats

if __name__ == '__main__':

    orig_data, _, _ = load_neuron_tx()

    np.random.seed(0)

    for shuffle in [True, False]:

        if shuffle:
            data = ax1_shuffle(orig_data)
        else:
            data = orig_data

        stats_dict = sz_compress_time_series(data, error_range=[1e-6])

        save_stats(stats_dict, get_parent_dir() / 'Mesh-Float-Zip/stats/sz' / f'sz_shuffle_{shuffle}_neuron_compression.pkl')

        print(f'Compression Ratios: {stats_dict["comp_ratio"]} and Frobenius Errors: {stats_dict["avg_rel_frob_error"]} and PSNR: {stats_dict["avg_psnr"]}')
