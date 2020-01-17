#!/usr/bin/env python3

"""
Load an example point cloud and show a local PCA
"""
import pyreg.functions
import numpy
from mayavi import mlab
from pyreg.point_cloud_class import *
import pyreg.visualization
import ipdb


def get_n_neighborhood(pcd, point, n):
    """ Find nearest neighbors in kD-tree search for each target point """
    if n == 0:
        return numpy.array([])
    else:
        nbrs = NearestNeighbors(n_neighbors=n, algorithm="auto", metric="euclidean", n_jobs=-1).fit(pcd.points)
        nn_indices = nbrs.kneighbors(point, return_distance=False)
        return nn_indices.flatten()

@mlab.show
def plot_pca(pcd, n_neighborhood, point, eigenvectors, eigen_values):
    fig1 = mlab.figure(bgcolor=(1,1,1))
    mean_nn = pyreg.functions.get_mean_nn_distance(pcd)
    pcd_pts = mlab.points3d(pcd.points[:, 0], pcd.points[:, 1], pcd.points[:, 2], 
                                color=(0, 0, 1), mode="sphere", figure=fig1, scale_factor=0.25*mean_nn)
    
    neighbors = mlab.points3d(n_neighborhood[:, 0], n_neighborhood[:, 1], n_neighborhood[:, 2], 
                                color=(1, 0, 0), mode="sphere", figure=fig1, scale_factor=0.5*mean_nn)
    
    pt = mlab.points3d(point[:, 0], point[:, 1], point[:, 2], 
                                color=(0, 0, 0), mode="sphere", figure=fig1, scale_factor=1*mean_nn)
    
    eigenvector1 = mlab.quiver3d(point[:, 0], point[:, 1], point[:, 2], 
                            eigenvectors[0, 0], eigenvectors[0, 1], eigenvectors[0, 2],
                            mode="arrow", color=(0, 0, 0), figure=fig1, scale_factor=3*mean_nn)
    eigenvector2 = mlab.quiver3d(point[:, 0], point[:, 1], point[:, 2], 
                            eigenvectors[1, 0], eigenvectors[1, 1], eigenvectors[1, 2],
                            mode="arrow", color=(0, 1, 0), figure=fig1, scale_factor=3*mean_nn)
    eigenvector3 = mlab.quiver3d(point[:, 0], point[:, 1], point[:, 2], 
                            eigenvectors[2, 0], eigenvectors[2, 1], eigenvectors[2, 2],
                            mode="arrow", color=(0, 1, 0), figure=fig1, scale_factor=3*mean_nn)
    
    return


@mlab.show
def plot_normal_in_neighborhood(pcd, point_idx, neighborhood):
    fig1 = mlab.figure(bgcolor=(1,1,1))
    mean_nn = pyreg.functions.get_mean_nn_distance(pcd)
    pcd_pts = mlab.points3d(pcd.points[:, 0], pcd.points[:, 1], pcd.points[:, 2], 
                                color=(0, 0, 1), mode="sphere", figure=fig1, scale_factor=0.25*mean_nn)
    
    neighbors = mlab.points3d(neighborhood[:, 0], neighborhood[:, 1], neighborhood[:, 2], 
                                color=(1, 0, 0), mode="sphere", figure=fig1, scale_factor=0.5*mean_nn)
    
    pt = mlab.points3d(pcd.points[point_idx, 0], pcd.points[point_idx, 1], pcd.points[point_idx, 2], 
                                color=(0, 0, 0), mode="sphere", figure=fig1, scale_factor=1*mean_nn)
    
    normal = mlab.quiver3d(pcd.points[point_idx, 0], pcd.points[point_idx, 1], pcd.points[point_idx, 2], 
                            pcd.normals[point_idx, 0], pcd.normals[point_idx, 1], pcd.normals[point_idx, 2],
                            mode="arrow", color=(0, 0, 0), figure=fig1, scale_factor=3*mean_nn)


def main():    
    pcd = PointCloud()
    pcd.read_pcd("pca.xyz")
    
    ipdb.set_trace()
    # TODO: Find index of good plot point
    idx = 10
    
    mean_nn = pcd.get_mean_nn_distance()
    pcd.estimate_normals_open3d(neighborhood=30, search_radius=20*mean_nn)
    
    point = pcd.points[idx].reshape((1, 3))
    n_idx = get_n_neighborhood(pcd, point, 30)
    indices = numpy.concatenate((numpy.array([idx]), n_idx))
    n_neighborhood = pcd.points[indices]
    
    #~ covariance_matrix = numpy.cov(n_neighborhood.T)
    #~ eigen_values, eigen_vectors = numpy.linalg.eig(covariance_matrix)
    
    #~ ipdb.set_trace()
    
    #~ sort_idx = eigen_values.argsort(axis=0) 
    #~ eigen_vectors = eigen_vectors[sort_idx]
    #~ eigen_values = eigen_values[sort_idx]
    
    #~ plot_pca(pcd, n_neighborhood, point, eigen_vectors, eigen_values)
    plot_normal_in_neighborhood(pcd, idx, n_neighborhood)
    
    

if __name__ == "__main__":
    main()

