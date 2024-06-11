import numpy as np
import matplotlib.pyplot as plt
from MFZ.block import Block


def run_test():
    
    np.random.seed(0)

    points = np.linspace(0,1,32).reshape(-1,1)

    data = np.convolve(np.random.normal(0,1,32), np.ones((16))/16, mode='same')

    B = Block(points)

    block_dict = B.compress(data, error_tol=1e-7)

    decoded = B.decompress(block_dict)

    assert np.allclose(data, decoded, atol=1e-3), "Decoded data does not match original data"

    print("Block Test Passed")




if __name__ == '__main__':
    run_test()