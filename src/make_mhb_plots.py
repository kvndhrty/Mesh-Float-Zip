from MFZ.plots import plot_mhb
import numpy as np

if __name__ == '__main__':

    points = ''

    for i in range(4):
        plot_mhb('ignition_mesh', channel=i, shuffle=True)
        plot_mhb('ignition_mesh', channel=i, shuffle=False)