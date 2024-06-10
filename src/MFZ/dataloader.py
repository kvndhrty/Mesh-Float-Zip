import numpy as np
import platform
from pathlib import Path
import meshio


def get_parent_dir():
    if platform.system() == 'Windows':
        parent_dir = Path(r"C:\Users\Kevin\OneDrive - UCB-O365\Documents\Research\GitHub\common-data")
    elif platform.system() == 'Darwin':
        parent_dir = Path(r"/Users/kdoh/Library/CloudStorage/OneDrive-UCB-O365/Documents/Research/GitHub/common-data")
    elif platform.system() == 'Linux':
        parent_dir = Path(r"/mnt/c/Users/Kevin/OneDrive - UCB-O365/Documents/Research/GitHub/common-data")
    return parent_dir


def peturb_data(data, noise=0.1):
    noise = np.random.normal(0, noise, data.shape)
    perturbed_data = data + noise
    return perturbed_data

def load_ignition_grid():
    parent_dir = get_parent_dir()
    file_path = parent_dir / "ignition_grid" / "features.npy"
    dataset = np.load(file_path).reshape(450,50,50,4)

    X,Y = np.meshgrid(np.linspace(0,1,50), np.linspace(0,1,50))
    points = np.stack([X,Y], axis=-1).reshape(50,50,2)

    return dataset, points


def load_ignition_mesh():
    parent_dir = get_parent_dir()
    points_path = parent_dir / "ignition_mesh" / "points.npy"
    data_path = parent_dir / "ignition_mesh" / "features.npy"

    points = np.load(points_path).reshape(2189, 2)
    dataset = np.load(data_path)

    return dataset, points


def load_flat_plate():
    parent_dir = get_parent_dir()
    points_path = parent_dir / "flat_plate_2d_cut" / "points_2.npy"
    data_path = parent_dir / "flat_plate_2d_cut" / "features_603.npy"

    points = np.load(points_path)
    dataset = np.load(data_path)

    return dataset, points




def load_neuron_tx():
    parent_dir = get_parent_dir()
    points_path = parent_dir / "Neuron Transport" / "NMO_54504" / "points.npy"
    data_path = parent_dir / "Neuron Transport" / "NMO_54504" / "concentration_all_fix.npy"
    mesh_path = parent_dir / "Neuron Transport" / "NMO_54504" / "mesh.vtk"


    points = np.load(points_path)
    dataset = np.load(data_path).swapaxes(0,1)

    mesh = meshio.read(mesh_path)

    return dataset, points, mesh