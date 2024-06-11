from MFZ.tests import mfz_compress_time_series
from MFZ.dataloader import load_neuron_tx
from pathlib import Path
from MFZ.util import save_stats
from MFZ.partition import mesh_to_adjacency_list

if __name__ == '__main__':

    data, points, mesh = load_neuron_tx()

    adj_list = mesh_to_adjacency_list(mesh)

    block_size = 512

    stats_dict = mfz_compress_time_series(data, points, error_range=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1/4, 1/2], block_size=block_size, adjacency_list=adj_list)

    save_stats(stats_dict, Path(r'C:\Users\Kevin\OneDrive - UCB-O365\Documents\Research\GitHub\Mesh-Float-Zip\stats\zfp') / f'block_{block_size}_neuron_compression.pkl')

    print(f'Compression Ratios: {stats_dict["comp_ratio"]} and Frobenius Errors: {stats_dict["avg_rel_frob_error"]} and PSNR: {stats_dict["avg_psnr"]} and block size {block_size}')

  

