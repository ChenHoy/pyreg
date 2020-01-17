#!/usr/bin/env python3

import numpy
import ipdb
import psutil
import time
import dask.array
from scipy.spatial import distance
import h5py
import os
from dask.diagnostics import ProgressBar
from dask.distributed import Client


def get_process_memory():
    process = psutil.Process(os.getpid())
    
    return process.memory_info().rss


def compute_sqeuclidean(a, b):
    """
    Compute the squared distance between every point of 2 dask point arrays
    
    input:
        a, b    = dask array
    """
    a_dim = a.ndim
    b_dim = b.ndim
    if a_dim == 1:
        a = a.dask.array(1, 1, a.shape[0])
    if a_dim >= 2:
        a = a.reshape(numpy.prod(a.shape[:-1]), 1, a.shape[-1])
    if b_dim > 2:
        b = b.reshape(numpy.prod(b.shape[:-1]), b.shape[-1])
    
    diff = a-b
    
    dist_arr = dask.array.einsum("ijk,ijk->ij", diff, diff)
    dist_arr = dask.array.squeeze(dist_arr)
    
    return dist_arr


def main():
    #~ client = Client() # start a local Dask client
    array1 = numpy.random.rand(100000, 3)
    dask1 = dask.array.from_array(array1, chunks=(1000, 3))
    
    array2 = numpy.random.rand(100000, 3)
    dask2 = dask.array.from_array(array2, chunks=(1000, 3))
    
    dist = compute_sqeuclidean(dask1, dask2)
    
    # Because we cannot compute this in our RAM we write out the computation to disk in hdf5 
    
    with ProgressBar():
        f = h5py.File("dist.hdf5")
        data = f.require_dataset("/data", shape=dist.shape, dtype=dist.dtype)
        dask.array.store(dist, data)
        
        #~ dask_sigma = dist.compute()
    
    f = h5py.File("myfile.hdf5")
    dset = f["/data"]
    test = dask.array.from_array(dset, chunks=(10, 3))
    
    

if __name__ == "__main__":
    main()    
