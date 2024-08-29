import numpy as np
import pymetis as metis
import scipy.spatial as spatial

from numba import jit, njit


def mesh_to_adjacency_list(mesh):
    adj = {}
    for cell in mesh.cells: # for each mesh structure

        for i in range(cell.data.shape[0]):  # for each hexagon
            
            for j in range(cell.data.shape[1]): # for each vertex

                for k in range(cell.data.shape[1] - j):

                    if cell.data[i,j] not in adj:
                        adj[cell.data[i,j] ] = []
                    
                    if cell.data[i,k] not in adj:
                        adj[cell.data[i,k]] = []

                    if cell.data[i,j] not in adj[cell.data[i,k]]:
                        adj[cell.data[i,k]].append(cell.data[i,j])

                    if cell.data[i,k] not in adj[cell.data[i,j]]:
                        adj[cell.data[i,j]].append(cell.data[i,k])
    return adj


def block_mesh(points, block_size_target=16, adjacency_list=None):

    num_blocks = np.ceil(len(points) / block_size_target).astype(np.int32)

    if adjacency_list is None:

        tri = spatial.Delaunay(points)

        adjacency_list = []

        (indptr, indices) = tri.vertex_neighbor_vertices

        for k in range(len(tri.points)):
            neighbors = indices[indptr[k]:indptr[k+1]]
            adjacency_list.append(np.array([n for n in neighbors if n != -1]))

    n_cuts, membership = metis.part_graph(adjacency=adjacency_list, nparts=num_blocks)

    return membership   


def block_tensor(tensor, block_size=4):
    """
    Breaks an arbitrary tensor into nxnxn blocks
    
    Args:
    tensor: numpy array
    
    Returns:
    numpy array of shape (n_blocks, block_size, block_size, block_size)
    """

    orig_shape = tensor.shape
    n_blocks = np.ceil(np.array(orig_shape) / block_size).astype(np.int32)

    padded_shape = list(nb * block_size for nb in n_blocks)
    padded_tensor = np.zeros(padded_shape)
    padded_tensor[:orig_shape[0], :orig_shape[1], :orig_shape[2]] = tensor

    blocks = block_inner_loop(padded_tensor, n_blocks, total_blocks=np.prod(n_blocks), block_size=block_size)

    return blocks

@njit
def block_inner_loop(padded_tensor, n_blocks, total_blocks, block_size=4):

    blocks = np.zeros((total_blocks, block_size, block_size, block_size))
    for i in range(n_blocks[0]):
        for j in range(n_blocks[1]):
            for k in range(n_blocks[2]):
                block = padded_tensor[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size, k*block_size:(k+1)*block_size]
                blocks[k*n_blocks[0]*n_blocks[1] + j*n_blocks[0] + i] = block
    return blocks


def unblock_tensor(blocks, orig_shape, block_size=4):
    """
    Reconstructs an arbitrary tensor from nxnxn blocks
    
    Args:
    blocks: numpy array of shape (n_blocks, block_size, block_size, block_size)
    orig_shape: tuple of ints, the original shape of the tensor
    
    Returns:
    numpy array of shape orig_shape
    """
    tensor = np.zeros(orig_shape)
    n_blocks = np.ceil(np.array(orig_shape) / block_size).astype(np.int32)

    tensor = unblock_inner_loop(blocks, tensor, orig_shape, n_blocks, block_size=block_size)

    return tensor



@njit
def unblock_inner_loop(blocks, tensor, orig_shape, n_blocks, block_size=4):
    for i in range(n_blocks[0]):
        for j in range(n_blocks[1]):
            for k in range(n_blocks[2]):
                block = blocks[k*n_blocks[0]*n_blocks[1] + j*n_blocks[0] + i]
                tensor[i*block_size:min((i+1)*block_size, orig_shape[0]), 
                        j*block_size:min((j+1)*block_size, orig_shape[1]), 
                        k*block_size:min((k+1)*block_size, orig_shape[2])] = block[:min(block_size, orig_shape[0]-i*block_size), 
                                                                                    :min(block_size, orig_shape[1]-j*block_size), 
                                                                                    :min(block_size, orig_shape[2]-k*block_size)]
    return tensor

@njit
def outer_product_basis(rows):
    """
    Builds an outer product basis for a n x n x n tensor using the rows of a matrix
    
    Args:
    rows: numpy array of shape (n, n)
    
    Returns:
    numpy array of shape (n**3, n, n, n)
    """
    n = rows.shape[0]
    basis = np.zeros((n**3, n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                basis[k*n**2 + i + j*n, :, :, :] = np.outer(rows[i, :] , np.outer(rows[j, :] , rows[k, :])).reshape(n,n,n)

    return basis

def test_block_unblock():
    dummy_data = np.random.rand(32,32,32)

    test_blocks = block_tensor(dummy_data)

    print(test_blocks.shape)

    test_tensor = unblock_tensor(test_blocks, dummy_data.shape)

    print(test_tensor.shape)

    np.allclose(test_tensor, dummy_data)

    print(np.linalg.norm(test_tensor - dummy_data))

    if np.allclose(test_tensor, dummy_data): print("Success!"); return True
    else: print("Failure!"); return False



