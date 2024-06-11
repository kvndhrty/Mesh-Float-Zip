from MFZ.tests import mfz_compress_time_series
from pathlib import Path
from MFZ.util import save_stats
from MFZ.dataloader import load_flat_plate

if __name__ == '__main__':

    data, points = load_flat_plate()

    block_size = 64

    for channel in range(data.shape[-1]):

        stats_dict = mfz_compress_time_series(data[:,:,channel], points, error_range=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1/4, 1/2], block_size=block_size)

        save_stats(stats_dict, Path(r'C:\Users\Kevin\OneDrive - UCB-O365\Documents\Research\GitHub\Mesh-Float-Zip\stats\zfp') / f'block_{block_size}_channel_{channel}_flat_plate_compression.pkl')

        print(f'Compression Ratios: {stats_dict["comp_ratio"]} and Frobenius Errors: {stats_dict["avg_rel_frob_error"]} and PSNR: {stats_dict["avg_psnr"]} for channel {channel} and block size {block_size}')
