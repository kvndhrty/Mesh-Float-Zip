from MFZ.tests import zfp_compress_time_series
from matplotlib import pyplot as plt

from MFZ.dataloader import load_ignition_mesh

if __name__ == '__main__':

    data, points = load_ignition_mesh()

    comp_ratio, frob_error = zfp_compress_time_series(data[:,:,1], error_range=[1e-3, 1e-2, 1e-1])

    print(f'Compression Ratios: {comp_ratio} and Frobenius Errors: {frob_error}')

    fig, ax = plt.subplots()

    plt.plot(frob_error, comp_ratio, 'o-')
    plt.xlabel('Frobenius Error')
    plt.ylabel('Compression Ratio')
    plt.title('ZFP Compression of Ignition Mesh')

    ax.set_xscale('log')

    plt.show()