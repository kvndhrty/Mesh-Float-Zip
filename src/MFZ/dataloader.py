import numpy as np
import platform
from pathlib import Path
import meshio


def get_parent_dir():
    if platform.system() == 'Windows':
        parent_dir = Path(r"C:\Users\Kevin\OneDrive - UCB-O365\Documents\Research\GitHub")
    elif platform.system() == 'Darwin':
        parent_dir = Path(r"/Users/kdoh/Library/CloudStorage/OneDrive-UCB-O365/Documents/Research/GitHub")
    elif platform.system() == 'Linux':
        parent_dir = Path(r"/mnt/c/Users/Kevin/OneDrive - UCB-O365/Documents/Research/GitHub")
    return parent_dir


def peturb_data(data, noise=0.1):
    noise = np.random.normal(0, noise, data.shape)
    perturbed_data = data + noise
    return perturbed_data

def shuffle_data(data, points):
    idx = np.random.permutation(data.shape[1])
    return data[:,idx], points[idx]

def ax1_shuffle(array):

    swapped = array.swapaxes(0,1)

    np.random.shuffle(swapped)

    de_swapped = swapped.swapaxes(0,1)

    return de_swapped

def load_ignition_grid():
    parent_dir = get_parent_dir()
    file_path = parent_dir / 'common-data' / "ignition_grid" / "features.npy"
    dataset = np.load(file_path).reshape(450,50,50,4)

    X,Y = np.meshgrid(np.linspace(0,1,50), np.linspace(0,1,50))
    points = np.stack([X,Y], axis=-1).reshape(50,50,2)

    return dataset, points


def load_ignition_mesh():
    parent_dir = get_parent_dir()
    points_path = parent_dir / 'common-data' / "ignition_mesh" / "points.npy"
    data_path = parent_dir / 'common-data' / "ignition_mesh" / "features.npy"

    points = np.load(points_path).reshape(2189, 2)
    dataset = np.load(data_path).astype(np.float32)

    return dataset, points


def load_ignition_grid_residual():
    parent_dir = get_parent_dir()
    data_path = parent_dir / 'common-data' / "ignition_grid_offline_residual" / "features.npy"
    dataset = np.load(data_path).reshape(450,2500,2)

    X,Y = np.meshgrid(np.linspace(0,1,50), np.linspace(0,1,50))
    points = np.stack([X,Y], axis=-1).reshape(2500,2)

    return dataset, points


def load_flat_plate():
    parent_dir = get_parent_dir()
    points_path = parent_dir / 'common-data' / "flat_plate_2d_cut" / "points_2.npy"
    data_path = parent_dir / 'common-data' / "flat_plate_2d_cut" / "features_603.npy"

    points = np.load(points_path)
    dataset = np.load(data_path).astype(np.float32)

    return dataset, points




def load_neuron_tx():
    parent_dir = get_parent_dir()
    points_path = parent_dir / 'common-data' / "Neuron Transport" / "NMO_54504" / "points.npy"
    data_path = parent_dir / 'common-data' / "Neuron Transport" / "NMO_54504" / "concentration_float32.npy"
    mesh_path = parent_dir / 'common-data' / "Neuron Transport" / "NMO_54504" / "mesh.vtk"

    mesh = meshio.read(mesh_path)
    points = np.load(points_path)
    dataset = np.load(data_path).astype(np.float32)

    

    return dataset, points, mesh