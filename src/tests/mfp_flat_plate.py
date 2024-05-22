from MFZ.tests import mfp_compress_time_series
from matplotlib import pyplot as plt

from MFZ.dataloader import load_flat_plate

if __name__ == '__main__':

    data, points = load_flat_plate()

    comp_ratio, frob_error = mfp_compress_time_series(data[:,:,2], points, error_range=[1e-4, 1e-3, 1e-2], block_size=32)

    print(f'Compression Ratios: {comp_ratio} and Frobenius Errors: {frob_error}')

    fig, ax = plt.subplots()

    plt.plot(frob_error, comp_ratio, 'o-')
    plt.xlabel('Frobenius Error')
    plt.ylabel('Compression Ratio')
    plt.title('MFP Compression of Flat Plate Data')

    ax.set_xscale('log')

    plt.show()