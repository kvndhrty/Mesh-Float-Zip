from MFZ.tests import sz_compress_time_series
import numpy as np
from MFZ.dataloader import load_flat_plate, shuffle_data, get_parent_dir
from MFZ.util import save_stats

if __name__ == '__main__':

    orig_data, orig_points = load_flat_plate()

    np.random.seed(0)

    for shuffle in [True, False]:

        data, points = shuffle_data(orig_data, orig_points) if shuffle_data else (orig_data, orig_points)

        for channel in range(data.shape[-1]):

            stats_dict = sz_compress_time_series(data[:,:,channel], error_range=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1/4, 1/2])

            save_stats(stats_dict, get_parent_dir() / 'Mesh-Float-Zip/stats/sz' / f'sz_shuffle_{shuffle}_flat_plate_channel_{channel}_compression.pkl')

            print(f'Compression Ratios: {stats_dict["comp_ratio"]} and Frobenius Errors: {stats_dict["avg_rel_frob_error"]} and PSNR: {stats_dict["avg_psnr"]} for channel {channel}')
