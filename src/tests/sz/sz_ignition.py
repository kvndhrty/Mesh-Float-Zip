from MFZ.tests import sz_compress_time_series
import numpy as np
from MFZ.dataloader import load_ignition_mesh

from pathlib import Path

from MFZ.util import save_stats

if __name__ == '__main__':

    data, points = load_ignition_mesh()

    np.random.seed(0)

    for channel in range(data.shape[-1]):

        stats_dict = sz_compress_time_series(data[:,:,channel], error_range=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1/4, 1/2])

        save_stats(stats_dict, Path(r'/Users/kdoh/Library/CloudStorage/OneDrive-UCB-O365/Documents/Research/GitHub/Mesh-Float-Zip/stats') / f'sz_ignition_mesh_channel_{channel}_compression.pkl')

        print(f'Compression Ratios: {stats_dict["comp_ratio"]} and Frobenius Errors: {stats_dict["avg_rel_frob_error"]} and PSNR: {stats_dict["avg_psnr"]} for channel {channel}')
