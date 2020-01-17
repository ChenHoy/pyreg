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
from pyreg.icpy.icp_registration import IcpRegistration
from pyreg.registration_base import ConvergenceCriteria
import pyreg.functions
import pyreg.visualization
from pyreg.point_cloud_class import *
from mayavi import mlab


@mlab.show
def draw_bounding_box(pcd):
    """ Draws scan (red) and target/model (green) in seperate window using mayavi visualizer (VTK) """
    pcd_temp = copy.deepcopy(pcd)
    fig = mlab.figure()
    pcd_pts = mlab.points3d(pcd_temp.points[:, 0], pcd_temp.points[:, 1], pcd_temp.points[:, 2], 
                                color=(0, 1, 0), mode="point")    
    pcd_bounding_box = mlab.points3d(pcd_temp.bounding_box[:, 0], pcd_temp.bounding_box[:, 1], pcd_temp.bounding_box[:, 2], 
                                color=(1, 0, 0), mode="sphere")
    mlab.outline()
    fig.scene.isometric_view()
    fig.scene.parallel_projection = True

    return


def compute_translation_from_bounding_box(pcd):
    """ We will only translate twice the bounding box shape """
    t_x = pcd.bounding_box[0, 0] - pcd.bounding_box[-1, 0]
    t_y = pcd.bounding_box[0, 1] - pcd.bounding_box[-1, 1]
    t_z = pcd.bounding_box[0, 2] - pcd.bounding_box[-1, 2]
    
    transformation = pyreg.functions.generate_transformation_zyx_euler_angles(0.0, 0.0, 0.0, 
                                                                              t_x, t_y, t_z)    
    
    return transformation

    
def downsample_pcd(pcd, voxel_size):
    """ Use voxel downsampling from Open3D on PointCloud object """
    pcd_open3d = open3d.PointCloud()
    pcd_open3d.points = open3d.Vector3dVector(pcd.points)
    
    pcd_open3d = open3d.voxel_down_sample(pcd_open3d, voxel_size=voxel_size)
    pcd = PointCloud(points=numpy.asarray(pcd_open3d.points))
    
    return pcd
    

def preprocessing(model, scan, k=2):
    """ 
    Necessary steps before registration:
        - Downsampling of both clouds 
            -> We want very similar densities!  
        - 
    """
    mean_nn_model = pyreg.functions.get_mean_nn_distance(model)
    mean_nn_scan = pyreg.functions.get_mean_nn_distance(scan)
    
    voxel_size = k*mean_nn_model
    downsample_pcd(model, voxel_size=voxel_size)
    downsample_pcd(scan, voxel_size=voxel_size)
    
    return model, scan    


def script_prequisites(script_dir, config):
    """ Prepare the logger and directory """
    timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
    log_file = "translation.log" # Make a logfile in current folder
    logger = pyreg.functions.create_timed_rotating_log(log_file) 
    logger.info("New Run \n ##############################################")
    
    return logger
    
    
def read_data(config, logger):
    """ Read in both clouds and center """
    target_raw = PointCloud()
    logger.info("Reading input point clouds...")
    target_raw.read_pcd("../../data/robustness/input/target/bunny_10_000.xyz")
    scan_raw = copy.deepcopy(target_raw)
    logger.info("Raw point cloud size: %d", target_raw.points.shape[0])
    scan_raw.center_around_origin()
    target_raw.center_around_origin()
    
    return target_raw, scan_raw
    

def main():
    """ ICP registration with point to point metric on Stanford Bunny """
    script_dir = os.getcwd()
    config = pyreg.functions.read_config("config.yml")
    logger = script_prequisites(script_dir, config)
    model_raw, scan_raw = read_data(config, logger)
    model, scan = preprocessing(model_raw, scan_raw, k=2)
    
    scan.get_bounding_box()
    translation = compute_translation_from_bounding_box(scan) 
    scan.transform(translation)
    rotation = pyreg.functions.generate_transformation_zyx_euler_angles(50.0, 0.0, 0.0, 
                                                                        0.0, 0.0, 0.0)
    scan.transform(rotation)
    pyreg.visualization.draw(model, scan)
    
    scan.center_around_origin()
    pyreg.visualization.draw(model, scan)
    
    rotation_2 = pyreg.functions.generate_transformation_zyx_euler_angles(-50.0, 0.0, 0.0, 
                                                                        0.0, 0.0, 0.0)
    scan.transform(rotation_2)
    pyreg.visualization.draw(model, scan)


if __name__ == "__main__":
    main()    

