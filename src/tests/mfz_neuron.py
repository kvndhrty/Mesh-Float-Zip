from MFZ.tests import mfz_compress_time_series, zfp_compress_time_series
from matplotlib import pyplot as plt

from MFZ.dataloader import load_neuron_tx

from MFZ.partition import mesh_to_adjacency_list

from pathlib import Path

if __name__ == '__main__':

    data, points, mesh = load_neuron_tx()

    adj_list = mesh_to_adjacency_list(mesh)

    block_size = 32

    comp_ratio, frob_error = mfz_compress_time_series(data, points, error_range=[1e-3], block_size=block_size, adjacency_list=adj_list)

    zcomp_ratio, zfrob_error = zfp_compress_time_series(data, error_range=[1e-3])

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

    plt.savefig( Path(r'C:\Users\Kevin\OneDrive - UCB-O365\Documents\Research\GitHub\Mesh-Float-Zip\figures') / f'block_{block_size}_neuron_tx_compression_zfp_bitstring.png')

    plt.show()

