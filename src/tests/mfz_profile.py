from MFZ.tests import mfz_compress_time_series
from MFZ.dataloader import load_neuron_tx
from MFZ.partition import mesh_to_adjacency_list

import cProfile

def run_compression():
    data, points, mesh = load_neuron_tx()
    adj_list = mesh_to_adjacency_list(mesh) 

    block_size = 128

    comp_ratio, frob_error = mfz_compress_time_series(data[0:1,:], points, error_range=[1e-2], block_size=block_size, adjacency_list=adj_list)

    print(f'Compression Ratios: {comp_ratio} and Frobenius Errors: {frob_error}')

    return comp_ratio, frob_error


if __name__ == '__main__':

    cProfile.run('run_compression()', sort='cumtime', filename='mfz_profile_neuron_tx')
