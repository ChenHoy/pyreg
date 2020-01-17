#!/usr/bin/env python3

"""
Visualize a point cloud with vtk or mayavi
"""

import os
import numpy 
import copy
import time
import pyreg.functions
import pyreg.visualization
import matplotlib
from pyreg.point_cloud_class import *
import scipy.spatial.distance
import scipy.stats
import ipdb 


def read_data(script_dir, config):
    """ Read in both clouds and center """
    target_raw = PointCloud()
    source_raw = PointCloud()
    target_raw.read_pcd(script_dir+"/"+config["files"]["target"])
    target_raw.center_around_origin()
    source_raw.read_pcd(script_dir+"/"+config["files"]["source"])
    source_raw.center_around_origin()
    
    return target_raw, source_raw


def extract_extreme_curvature_pcd(pcd, quartile=0.8):
    """ Get all extreme curvature values and create new point cloud with them """
    mean = numpy.mean(pcd.curvature.flatten())
    std = numpy.std(pcd.curvature.flatten())
    confid_interval = scipy.stats.norm.interval(quartile, loc=mean, scale=std)
    idx = numpy.where((pcd.curvature.flatten() < confid_interval[0]) | (pcd.curvature.flatten() > confid_interval[1]))[0]
    
    new_pcd = PointCloud(points=pcd.points[idx, :],
                        normals=pcd.normals[idx, :],
                        curvature=pcd.curvature[idx]
                        )
    
    return new_pcd


def main():
    """ Short script for visualizing the registration process """
    script_dir = os.getcwd()
    config = pyreg.functions.read_config("config.yml")
    model_raw, scan_raw = read_data(script_dir, config)
    
    ipdb.set_trace()
    
    path = script_dir+"/"+os.path.split(config["files"]["source"])[0]
    scan_raw.write_pcd(file_format="ply", filepath=path+"/OuterSheet_sampled_registered_filtered.ply")
    
    
    
    
if __name__ == "__main__":
    main()
