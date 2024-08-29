

import numpy as np
import scipy.linalg as la

from jax import grad, jacrev, vmap

import jax.numpy as jnp

from rbffd import *

from rbffd.rbf_fd import c_trace




def make_mhb(points, epsilon=1.0):


    # Build the matrices with the RBF centers and evaluation points
    N_tot = points.shape[0]

    D_X = np.tile(np.expand_dims(points, axis=0), (N_tot, 1, 1))

    D_Y = D_X.transpose((1,0,2))


    # RBF function and vectorized version
    rbf = lambda x, y : gaussian_rbf(x,y,epsilon)

    vec_rbf = vmap(rbf, 0 , 0 )

    vmap_rbf = lambda arg1, arg2 : vmap(vec_rbf, 1, 1)(arg1, arg2)


    # Laplacian of RBF function and vectorized version
    laplacian_rbf = lambda x, y : c_trace(jacrev(grad(rbf, argnums=(0)))(x, y))

    vec_laplacian_rbf = vmap(laplacian_rbf, 0, 0)

    vmap_laplacian_rbf = lambda arg1, arg2 : vmap(vec_laplacian_rbf, 1, 1)(arg1, arg2)


    # Build the RBF interpolation matrix (A) and the function laplacian matrix (B)
    A = vmap_rbf(D_X, D_Y) + 1e-3*jnp.eye(N=N_tot)
    B = vmap_laplacian_rbf(D_X, D_Y)

    eigenvalues, eigenvectors = la.eigh(np.array(B, dtype=np.float64), np.array(A, dtype=np.float64))

    basis = A @ eigenvectors

    basis = np.array(basis)

    basis = basis / np.linalg.norm(basis, axis=0, ord=2)

    return basis


import spharapy.trimesh as tm
import spharapy.spharabasis as sb

import scipy.spatial as spatial


def make_sphara_mhb(points,my_mesh=None):

    if my_mesh is None:

        tri = spatial.Delaunay(points)

        trilist = tri.simplices

        if points.shape[1] == 2:
            points = np.hstack((points, np.zeros((points.shape[0], 1))))

        my_mesh = tm.TriMesh(trilist, points)

    sphara_basis = sb.SpharaBasis(my_mesh, 'fem')
    basis_functions, natural_frequencies = sphara_basis.basis()

    basis_functions = basis_functions / np.linalg.norm(basis_functions, axis=0, ord=2)

    return basis_functions


def make_random_basis(points,):

    basis_functions = np.random.randn(points.shape[0], points.shape[0])

    Q, R = la.qr(basis_functions)

    return Q