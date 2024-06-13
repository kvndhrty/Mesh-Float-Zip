from MFZ.tests import zfp_compress_time_series
import numpy as np
from MFZ.dataloader import load_neuron_tx, get_parent_dir, ax1_shuffle
from MFZ.util import save_stats
import copy

if __name__ == '__main__':

    orig_data, orig_points, _ = load_neuron_tx()

    np.random.seed(0)

    for shuffle in [True, False]:

        if shuffle:
            data = ax1_shuffle(copy.deepcopy(orig_data))
        else:
            data = orig_data

        stats_dict = zfp_compress_time_series(data, error_range=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1/4, 1/2])

        save_stats(stats_dict, get_parent_dir() / 'Mesh-Float-Zip/stats/zfp' / f'zfp_shuffle_{shuffle}_neuron_compression.pkl')

        print(f'Compression Ratios: {stats_dict["comp_ratio"]} and Frobenius Errors: {stats_dict["avg_rel_frob_error"]} and PSNR: {stats_dict["avg_psnr"]}')
