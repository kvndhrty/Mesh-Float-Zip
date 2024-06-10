import numpy as np

from MFZ.partition import block_tensor, unblock_tensor, outer_product_basis

from numba import jit, njit

@njit
def tensor_basis_transform(data, basis):

    spectrum = np.zeros_like(data).reshape(data.shape[0], -1)

    for j in range(data.shape[0]):
        for i in range(basis.shape[0]):
            spectrum[j, i] = np.sum(data[j, :, :, :] * basis[i, :, :, :])

    return spectrum.reshape(*data.shape)


def transform_time_series(dataset, inverse=False, Q=np.eye(4), block_size=4):

    if not inverse:
        op_basis = outer_product_basis(Q)
    elif inverse:
        op_basis = outer_product_basis(np.linalg.inv(Q))

    dataset_spectrum = np.zeros_like(dataset)

    for k in range(dataset.shape[0]):

        temp_blocks = block_tensor(dataset[k], block_size=block_size)

        dataset_spectrum[k, :, :, :] = unblock_tensor(tensor_basis_transform(temp_blocks, op_basis), dataset[k].shape, block_size=block_size)

    return dataset_spectrum


def zfp_dct_basis(t = (2/np.pi) * np.arcsin(1/(2*np.sqrt(2)))):

    Q = np.zeros((4,4))

    s = np.sqrt(2) * np.sin(t*np.pi/2)

    c = np.sqrt(2) * np.cos(t*np.pi/2)

    Q[0,:] = [1, 1, 1, 1]
    Q[1,:] = [c, s, -s, -c]
    Q[2, :] = [1, -1, -1, 1]
    Q[3, :] = [s, -c, c, -s]

    return Q * 1/2


import pickle
from pathlib import Path

def save_stats(stats : dict, filename : str):
    # saves a dictionary of information to a file

    filename = Path(filename)

    with filename.open('wb') as f:
        pickle.dump(stats, f)


def load_stats(filename : str):
    # loads a dictionary of information from a file

    filename = Path(filename)

    with filename.open('rb') as f:
        stats = pickle.load(f)

    return stats


def psnr(data, recon):

    data_spread = np.max(data) - np.min(data)

    data = data.reshape(recon.shape)

    return 10 * np.log10(data_spread**2 / np.mean((data - recon)**2))


def relative_frob(data, recon):

    data = data.reshape(recon.shape)

    rb = np.sqrt(np.sum((data - recon)**2)) / (np.sqrt(np.sum((data)**2)) + 1e-6)

    return rb