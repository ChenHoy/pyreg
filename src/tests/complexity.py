#!/usr/bin/env python3

import numpy 
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    """ Plot the complexity of the correspondence search """
    M = numpy.arange(1000)
    N = numpy.arange(1000)
    X, Y = numpy.meshgrid(N, M)
    brute_force = X*Y
    kd_tree = (X+Y)*numpy.log2(X+1)
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection="3d")
    ax1.plot_surface(X, Y, brute_force,
                       antialiased=False, label="Brute Force")
    plt.title("Brute Force", fontsize=15)
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2.plot_surface(X, Y, kd_tree,
                       antialiased=False, label="kd trees")
    plt.title("kD-tree", fontsize=15)
    
    
    plt.show()
    


if __name__ == "__main__":
    main()    


