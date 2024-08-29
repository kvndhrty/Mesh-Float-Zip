from MFZ.plots import plot_stats


if __name__ == '__main__':

    plot_stats('neuron', channel=None, shuffle=True)
    plot_stats('neuron', channel=None, shuffle=False)


    #for i in range(4):
    #    plot_stats('flat_plate', channel=i, shuffle=True)
    #    plot_stats('flat_plate', channel=i, shuffle=False)