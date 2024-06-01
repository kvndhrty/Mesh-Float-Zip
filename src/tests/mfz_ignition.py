from MFZ.tests import mfz_compress_time_series, zfp_compress_time_series
from matplotlib import pyplot as plt
import numpy as np
from MFZ.dataloader import load_ignition_mesh

from pathlib import Path

if __name__ == '__main__':

    data, points = load_ignition_mesh()

    np.random.seed(0)

    block_size = 16

    comp_ratio, frob_error = mfz_compress_time_series(data[:,:,1], points, error_range=[1e-3, 1e-2, 1e-1], block_size=block_size)

    data_swapped = data.swapaxes(0,1)

    np.random.shuffle(data_swapped)

    re_shuffled_data = data_swapped.swapaxes(0,1)

    zcomp_ratio, zfrob_error = zfp_compress_time_series(re_shuffled_data[:,:,1], error_range=[1e-3, 1e-2, 1e-1])

    print(f'Compression Ratios: {comp_ratio} and Frobenius Errors: {frob_error}')
    print(f'Compression Ratios: {zcomp_ratio} and Frobenius Errors: {zfrob_error}')

    fig, ax = plt.subplots()

    plt.plot(zfrob_error, zcomp_ratio, 'o-', label='ZFP')
    plt.xlabel('Frobenius Error')
    plt.ylabel('Compression Ratio')
    plt.title(f'MFZ vs ZFP Compression of Ignition Mesh: Block Size {block_size}')


    plt.plot(frob_error, comp_ratio, 'k--', label='MFZ')
    plt.legend()

    ax.set_xscale('log')

    plt.savefig( Path(r'C:\Users\Kevin\OneDrive - UCB-O365\Documents\Research\GitHub\Mesh-Float-Zip\figures') / f'block_{block_size}_ignition_mesh_compression_shuffle.png')

    plt.show()

