#!/usr/bin/env python3

"""
Compute matchings of a transformed set via the ICP
"""

import os
import numpy 
import ipdb
import copy
import time
import open3d
from sklearn.neighbors import NearestNeighbors
from pyreg.icpy.icp_registration import IcpRegistration
from pyreg.registration_base import ConvergenceCriteria
import pyreg.functions
#~ import pyreg.visualization
from pyreg.point_cloud_class import *
from line_profiler import LineProfiler  


def estimate_normals_open3d(pcd, k=1):
    dummy = copy.deepcopy(pcd)
    mean_nn = pyreg.functions.get_mean_nn_distance(dummy)
    voxel_size = k*mean_nn

    pcd_open3d = open3d.PointCloud()
    pcd_open3d.points = open3d.Vector3dVector(dummy.points)
    
    pcd_open3d = open3d.voxel_down_sample(pcd_open3d, voxel_size = voxel_size)
    
    radius_normal = 2*voxel_size
    open3d.estimate_normals(pcd_open3d, open3d.KDTreeSearchParamHybrid(
                                            radius = radius_normal, max_nn = 30)
                            )

    pcd = PointCloud(points=numpy.asarray(pcd_open3d.points), 
                     normals=numpy.asarray(pcd_open3d.normals))

    return pcd
    
    
def estimate_normals_python(pcd, k=1):
    dummy = copy.deepcopy(pcd)
    mean_nn = pyreg.functions.get_mean_nn_distance(dummy)
    voxel_size = k*mean_nn

    pcd_open3d = open3d.PointCloud()
    pcd_open3d.points = open3d.Vector3dVector(dummy.points)
    
    pcd_open3d = open3d.voxel_down_sample(pcd_open3d, voxel_size = voxel_size)

    pcd = PointCloud(points=numpy.asarray(pcd_open3d.points))
    
    radius_normal = 2*voxel_size
    pcd.estimate_normals_and_curvature(neighborhood=30, search_radius=radius_normal)

    return pcd
    
    
def estimate_normals_python_new(pcd, k=1):
    dummy = copy.deepcopy(pcd)
    mean_nn = pyreg.functions.get_mean_nn_distance(dummy)
    voxel_size = k*mean_nn

    pcd_open3d = open3d.PointCloud()
    pcd_open3d.points = open3d.Vector3dVector(dummy.points)
    
    pcd_open3d = open3d.voxel_down_sample(pcd_open3d, voxel_size = voxel_size)

    pcd = PointCloud(points=numpy.asarray(pcd_open3d.points))
    
    radius_normal = 2*voxel_size
    pcd = compute_normals_new(pcd, neighborhood=30, search_radius=radius_normal)

    return pcd
    
    
def get_line_profile(function, *args):
    """ Print out the line_profiler for a function to test with given args """
    lp = LineProfiler()
    lp_wrapper = lp(function)
    lp_wrapper(*args)
    lp.print_stats()
    
    
def compute_normals_new(pcd, neighborhood, search_radius):
        """ 
        Enhance performance of normal estimation
        """
        pcd.normals = numpy.zeros((pcd.points.shape[0], pcd.points.shape[1]))
        
        for idx, point in enumerate(pcd.points):
            point = point.reshape((1, -1))
            nbrs = NearestNeighbors(n_neighbors=neighborhood+1, algorithm="kd_tree", metric="euclidean", radius=search_radius).fit(pcd.points)
            nn_distances, nn_indices = nbrs.radius_neighbors(point, return_distance=True)
            dist, ind = nn_distances[0], nn_indices[0]
            dist, ind = numpy.delete(dist, 0), numpy.delete(ind, 0) # Delete point to get only neighbors
            
            ind = dist.argsort(axis=0) # Sort neighbors by smallest distance
            
            if ind.shape[0] > neighborhood:
                ind = numpy.delete(ind, numpy.arange(neighborhood, ind.shape[0]))
            else:
                pass
            nearest_neighbors = pcd.points[ind]
            
            covariance_matrix = numpy.cov(nearest_neighbors.T)
            
            eigen_values, eigen_vectors  = numpy.linalg.eig(covariance_matrix)
            min_idx = eigen_values.argmin(axis=0)
            pcd.normals[idx] = eigen_vectors[min_idx]
    
        return pcd
    

def main():
    """ Compare optimized python code to the open3d implementation """
    script_dir = os.getcwd()
    pcd = PointCloud()
    try:
        pcd.read_pcd("../../data/noise/input/target/bunny_1000.xyz")
    except Exception:
        raise Exception("Failed to load file!")
    
    #~ pcd = estimate_normals_open3d(pcd, k=2)
    #~ get_line_profile(estimate_normals_open3d, pcd)
    
    #~ pcd = estimate_normals_python(pcd, k=2)
    #~ get_line_profile(estimate_normals_python, pcd)
    
    #~ ipdb.set_trace()
    
    


if __name__ == "__main__":
    main()
