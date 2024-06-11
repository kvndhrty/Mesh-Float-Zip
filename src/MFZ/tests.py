from MFZ.mfz import MFZ
import numpy as np
import zfpy as zfp

from pathlib import Path

from pysz.pysz import SZ
import sys

from tqdm import tqdm

from MFZ.util import relative_frob, psnr


def mfz_compress_time_series(data, points, error_range=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1], block_size=16, epsilon=1.0, adjacency_list=None):

    orig_data_total_bits = data.shape[1] * 32 * data.shape[0]

    stats_dict = {'avg_rel_frob_error': [], 'avg_psnr' : [],  'comp_ratio': [], 'error_tol': [], 'block_size': block_size, 'epsilon': epsilon}

    for error_tol in tqdm(error_range, desc='Error Tolerance'):

        compressor = MFZ(points=points, error_tol=error_tol, block_size=block_size, epsilon=epsilon, adjacency_list=adjacency_list)

        total_bits = 0

        frob_avg = 0

        psnr_avg = 0

        for i in tqdm(range(data.shape[0]), desc='Time Step', leave=False):

            compressed_data, bit_count = compressor.compress(data[i,:])

            total_bits += bit_count

            decomp_data = compressor.decompress(compressed_data)

            frob_avg += relative_frob(data[i,:], decomp_data)

            psnr_avg += psnr(data[i,:], decomp_data)

        stats_dict['comp_ratio'].append(orig_data_total_bits / total_bits)
        stats_dict['avg_rel_frob_error'].append(frob_avg / data.shape[0])
        stats_dict['avg_psnr'].append(psnr_avg / data.shape[0])
        stats_dict['error_tol'].append(error_tol)

    return stats_dict





def zfp_compress_time_series(data, error_range=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1]):

    stats_dict = {'avg_rel_frob_error': [], 'avg_psnr' : [],  'comp_ratio': [], 'error_tol': []}

    orig_data_total_bytes = data.shape[0] * data.shape[1] * 32/8

    for error_tol in tqdm(error_range, desc='Error Tolerance'):

        frob_avg = 0

        psnr_avg = 0

        total_bytes = 0

        for i in tqdm(range(data.shape[0]), desc='Time Step', leave=False):

            compressed_data = zfp.compress_numpy(data[i,:], tolerance=error_tol)

            total_bytes += len(compressed_data)

            decomp_data = zfp.decompress_numpy(compressed_data)

            frob_avg += relative_frob(data[i,:], decomp_data)

            psnr_avg += psnr(data[i,:], decomp_data)

        stats_dict['comp_ratio'].append(orig_data_total_bytes / total_bytes)
        stats_dict['avg_rel_frob_error'].append(frob_avg / data.shape[0])
        stats_dict['avg_psnr'].append(psnr_avg / data.shape[0])
        stats_dict['error_tol'].append(error_tol)
        
    return stats_dict



def sz_compress_time_series(data, error_range=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1]):

    stats_dict = {'avg_rel_frob_error': [], 'avg_psnr' : [],  'comp_ratio': [], 'error_tol': []}

    if sys.platform != 'darwin':
        ValueError('SZ3 tests are only supported on MacOS')

    lib_extention = {
    "darwin": "libSZ3c.dylib",
    "win32": "SZ3c.dll",
    }.get(sys.platform, "libSZ3c.so")

    data = data.astype(np.float32)

    sz_dylib = Path(r'/Users/kdoh/Library/CloudStorage/OneDrive-UCB-O365/Documents/Research/GitHub/SZ3/build_darwin/lib') / lib_extention

    sz = SZ(str(sz_dylib))

    for error_tol in tqdm(error_range, desc='Error Tolerance'):

        frob_avg = 0

        psnr_avg = 0

        comp_rat_avg = 0

        for i in tqdm(range(data.shape[0]), desc='Time Step', leave=False):

            compressed_data, cmpr_ratio = sz.compress(data[i,:], 0, error_tol, 0, 0)

            decomp_data = sz.decompress(compressed_data, data[i,:].shape, data.dtype)

            comp_rat_avg += cmpr_ratio

            frob_avg += relative_frob(data[i,:], decomp_data)

            psnr_avg += psnr(data[i,:], decomp_data)

        stats_dict['comp_ratio'].append(comp_rat_avg / data.shape[0])
        stats_dict['avg_rel_frob_error'].append(frob_avg / data.shape[0])
        stats_dict['avg_psnr'].append(psnr_avg / data.shape[0])
        stats_dict['error_tol'].append(error_tol)
        
    return stats_dict