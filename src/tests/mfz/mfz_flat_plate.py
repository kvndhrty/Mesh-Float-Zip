from MFZ.tests import mfz_compress_time_series
import numpy as np
from MFZ.util import save_stats
from MFZ.dataloader import load_flat_plate, shuffle_data, get_parent_dir

if __name__ == '__main__':

    orig_data, orig_points = load_flat_plate()

    np.random.seed(0)

    block_size = 64

    for shuffle in [False]:

        data, points = shuffle_data(orig_data, orig_points) if shuffle_data else (orig_data, orig_points)

        for channel in range(data.shape[-1]):

            stats_dict = mfz_compress_time_series(data[:,:,channel], points, error_range=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1/4, 1/2], block_size=block_size)

            save_stats(stats_dict, get_parent_dir() / 'Mesh-Float-Zip\stats\mfz' / f'mfz_shuffle_{shuffle}_block_{block_size}_channel_{channel}_flat_plate_compression.pkl')

            print(f'Compression Ratios: {stats_dict["comp_ratio"]} and Frobenius Errors: {stats_dict["avg_rel_frob_error"]} and PSNR: {stats_dict["avg_psnr"]} for channel {channel} and block size {block_size}')
