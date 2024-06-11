from MFZ.tests import sz_compress_time_series
import numpy as np
from MFZ.dataloader import load_neuron_tx

from pathlib import Path

from MFZ.util import save_stats

# THIS DOES NOT FUNCTION DUE TO A BUG IN SZ3

if __name__ == '__main__':

    data, points, mesh = load_neuron_tx()

    np.random.seed(0)

    stats_dict = sz_compress_time_series(data, error_range=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1/4, 1/2])

    #save_stats(stats_dict, Path(r'/Users/kdoh/Library/CloudStorage/OneDrive-UCB-O365/Documents/Research/GitHub/Mesh-Float-Zip/stats') / f'sz_neuron_compression.pkl')

    print(f'Compression Ratios: {stats_dict["comp_ratio"]} and Frobenius Errors: {stats_dict["avg_rel_frob_error"]} and PSNR: {stats_dict["avg_psnr"]}')
