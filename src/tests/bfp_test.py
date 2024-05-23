import numpy as np
import matplotlib.pyplot as plt

from MFZ.bfp import BFP


def run_test():
    
    np.random.seed(0)

    data = np.convolve(np.random.normal(0,1,32), np.ones((2))/2, mode='same')

    bfp = BFP(data, error_tol=1e-3)

    decoded = bfp.float()

    assert np.allclose(data, decoded, atol=1e-3), "Decoded data does not match original data"




if __name__ == '__main__':
    run_test()