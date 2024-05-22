import numpy as np

from MFZ.partition import block_mesh

from MFZ.block import Block




class MFP(object):

    def __init__(self, points, error_tol=1e-3, block_size=4, epsilon = 1.0):

        assert isinstance(points, np.ndarray), "points must be a numpy array"

        assert points.ndim == 2, "points must be a 2D array"

        self.points = points
        self.error_tol = error_tol
        self.block_size = block_size
        self.block_list = []

        self.epsilon = epsilon

        self.membership = block_mesh(self.points, self.block_size)

        self.num_blocks = max(self.membership) + 1

        self._init_blocks()

        return
    
    def _get_block(self, k):

        nodes = np.argwhere(np.array(self.membership) == k).ravel()

        return nodes

    def _init_blocks(self):

        for k in range(self.num_blocks):
            nodes = self._get_block(k)

            self.block_list.append(Block(self.points[nodes], self.epsilon))

        return
    
    def compress(self, data):

        cmprssd = []

        total_bits = 0
        
        for k in range(self.num_blocks):
            
            block = self.block_list[k]

            nodes = self._get_block(k)

            cmprssd_block = block.compress(data[nodes], self.error_tol)
            
            cmprssd.append(cmprssd_block)

            total_bits += block.count_bits(cmprssd_block)

        return cmprssd, total_bits
    
    def decompress(self, cmprssd):

        decomp_data = np.zeros((self.points.shape[0],1))

        for k in range(self.num_blocks):

            block = self.block_list[k]

            nodes = self._get_block(k)

            decomp_data[nodes] = block.decompress(cmprssd[k]).reshape(-1,1)

        return decomp_data


            
        

