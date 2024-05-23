from MFZ.tests import mfz_compress_time_series
from matplotlib import pyplot as plt

from MFZ.dataloader import load_ignition_mesh

if __name__ == '__main__':

    data, points = load_ignition_mesh()

    block_size = 64

    comp_ratio, frob_error = mfz_compress_time_series(data[:,:,1], points, error_range=[1e-4, 1e-3, 1e-2], block_size=block_size)

    print(f'Compression Ratios: {comp_ratio} and Frobenius Errors: {frob_error}')

    fig, ax = plt.subplots()

    plt.plot(frob_error, comp_ratio, 'o-')
    plt.xlabel('Frobenius Error')
    plt.ylabel('Compression Ratio')
    plt.title(f'MFP Compression of Ignition Mesh: Block Size {block_size}')

    ax.set_xscale('log')

    plt.show()