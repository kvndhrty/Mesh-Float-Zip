

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
    A = vmap_rbf(D_X, D_Y) + 0.1*jnp.eye(N=N_tot) 
    B = vmap_laplacian_rbf(D_X, D_Y)

    #A_row_sum = jnp.sum(A, axis=1) - jnp.diag(A)

    #A = A + jnp.diag(A_row_sum)


    eigenvalues, eigenvectors = la.eigh(B, A)

    chol_A = la.cholesky(A)

    return chol_A @ eigenvectors
