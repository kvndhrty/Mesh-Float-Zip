from MFZ.mfz import MFZ
import numpy as np
import zfpy as zfp

from pathlib import Path

from pysz.pysz import SZ
import sys

from tqdm import tqdm

from MFZ.util import relative_frob, psnr

import dask


def dask_mfz_compress_time_series(data, points, error_range=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1], block_size=16, epsilon=1.0, adjacency_list=None):

    orig_data_total_bits = data.shape[1] * 32 * data.shape[0]

    comp_ratio = []

    frob_error = []

    for error_tol in tqdm(error_range, desc='Error Tolerance'):

        compressor = MFZ(points=points, error_tol=error_tol, block_size=block_size, epsilon=epsilon, adjacency_list=adjacency_list)

        #total_bits = 0

        frob_avg = 0

        tasks = []

        for i in tqdm(range(data.shape[0]), desc='Compression: Time Step', leave=False):

            tasks.append( compressor.dask_compress(data[i,:]) ) 

            #total_bits += bit_count

            #decomp_data = compressor.decompress(compressed_data)

            #frob_avg = np.sqrt(np.sum((data[i,:].reshape(-1,1) - decomp_data)**2)) / (np.sqrt(np.sum((data[i,:].reshape(-1,1))**2)) + 1e-6)

        #comp_ratio.append(orig_data_total_bits / total_bits)
        #frob_error.append(frob_avg / data.shape[0])

        tasks = dask.compute(tasks)

    return tasks