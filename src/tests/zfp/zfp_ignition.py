from MFZ.tests import zfp_compress_time_series
from MFZ.util import save_stats
from pathlib import Path
from MFZ.dataloader import load_ignition_mesh

if __name__ == '__main__':

    data, points = load_ignition_mesh()

    for channel in range(data.shape[-1]):

        stats_dict = zfp_compress_time_series(data[:,:,channel], error_range=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1/4, 1/2])

        save_stats(stats_dict, Path(r'/Users/kdoh/Library/CloudStorage/OneDrive-UCB-O365/Documents/Research/GitHub/Mesh-Float-Zip/stats/zfp') / f'zfp_ignition_mesh_channel_{channel}_compression.pkl')

        print(f'Compression Ratios: {stats_dict["comp_ratio"]} and Frobenius Errors: {stats_dict["avg_rel_frob_error"]} and PSNR: {stats_dict["avg_psnr"]}')
