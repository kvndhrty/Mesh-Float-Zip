from MFZ.mfp import MFP
import numpy as np
import zfpy as zfp
import matplotlib.pyplot as plt 




def mfp_compress_time_series(data, points, error_range=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1], block_size=16, epsilon=1.0):

    orig_data_total_bits = data.shape[1] * 32 * data.shape[0]

    comp_ratio = []

    frob_error = []

    for error_tol in error_range:

        compressor = MFP(points=points, error_tol=error_tol, block_size=block_size, epsilon=epsilon)

        total_bits = 0

        frob_avg = 0

        for i in range(data.shape[0]):

            compressed_data, bit_count = compressor.compress(data[i,:])

            total_bits += bit_count

            decomp_data = compressor.decompress(compressed_data)

            frob_avg = np.sqrt(np.sum((data[i,:].reshape(-1,1) - decomp_data)**2)) / (np.sqrt(np.sum((data[i,:].reshape(-1,1))**2)) + 1e-6)

        comp_ratio.append(orig_data_total_bits / total_bits)
        frob_error.append(frob_avg / data.shape[0])

    return comp_ratio, frob_error





def zfp_compress_time_series(data, error_range=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1]):

    orig_data_total_bytes = data.shape[0] * data.shape[1] * 32/8

    comp_ratio = []

    frob_error = []

    for error_tol in error_range:

        total_bytes = 0

        frob_avg = 0

        for i in range(data.shape[0]):

            compressed_data = zfp.compress_numpy(data[i,:], tolerance=error_tol)

            total_bytes += len(compressed_data)

            decomp_data = zfp.decompress_numpy(compressed_data)

            frob_avg = np.sqrt(np.sum((data[i,:] - decomp_data)**2)) / np.sqrt(np.sum((data[i,:].reshape(-1,1))**2) + 1e-6)

        comp_ratio.append(orig_data_total_bytes / total_bytes)
        frob_error.append(frob_avg / data.shape[0])
        
    return comp_ratio, frob_error