#!/usr/bin/env python3

"""
SAC-IA from Open3d registration
"""

import os
import numpy 
import subprocess
import re
import ipdb
import copy
import time
from pyreg.global_reg.open3d_fast_bcu import Open3dFastBCURegistration
from pyreg.local_reg.local_registration_base import ConvergenceCriteria
import pyreg.functions
import pyreg.visualization
from pyreg.point_cloud_class import *


def script_prequisites(script_dir, config):
    """ Prepare the logger and directory """
    if config["files"]["save"] == True:
        timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
        output_path = config["files"]["output"]+"icp_"+timestr
        pyreg.functions.make_dir(output_path)
        pyreg.functions.write_config(config, output_path)
        log_file = output_path + "/global.log"
        logger = pyreg.functions.create_timed_rotating_log(log_file)
        logger.info("New Run \n ##############################################")
        
        return output_path, logger
    
    else:
        timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
        log_file = "global.log" # Make a logfile in current folder
        logger = pyreg.functions.create_timed_rotating_log(log_file) 
        logger.info("New Run \n ##############################################")
        
        return logger


def read_data(script_dir, config, logger):
    """ Read in both clouds and center """
    target_raw = PointCloud()
    source_raw = PointCloud()
    logger.info("Reading input point clouds...")    
    target_raw.read_pcd(script_dir+"/"+config["files"]["target"])
    target_raw.center_around_origin()
    source_raw.read_pcd(script_dir+"/"+config["files"]["source"])
    source_raw.center_around_origin()
    
    # TEMP
    ground_truth = pyreg.functions.generate_transformation_zyx_euler_angles(130.0, -40.0, 20.0, 0.0, 0.0, 0.0)
    source_raw.transform(ground_truth)
    
    logger.info("Source point cloud size: %d", source_raw.points.shape[0])
    logger.info("Target point cloud size: %d", target_raw.points.shape[0])
    
    return target_raw, source_raw
  
 
def refinement(scan, target, config, logger, trans_init=numpy.eye(4), overlap=1.0):
    """ Refine the coarse result with ICP """
    if config["icp_parameters"]["error_metric"] == "point_to_plane":
        logger.info("Estimating normals...")
        start = time.time()
        mean_nn = pyreg.functions.get_mean_nn_distance(target)
        scan.estimate_normals_open3d(neighborhood=config["icp_parameters"]["estimate_normals"]["max_nn"], 
                                     search_radius=config["icp_parameters"]["estimate_normals"]["search_radius"]*mean_nn)
        target.estimate_normals_open3d(neighborhood=config["icp_parameters"]["estimate_normals"]["max_nn"], 
                                     search_radius=config["icp_parameters"]["estimate_normals"]["search_radius"]*mean_nn)
        end = time.time()
        time_normals = end-start
    else:
        pass
    
    reg = IcpRegistration(copy.deepcopy(scan), copy.deepcopy(target), trans_init, config, logger, 
                            ConvergenceCriteria(max_iteration=config["convergence_criteria"]["max_iteration"],
                                                tolerance=config["convergence_criteria"]["relative_rmse"],
                                                error_metric=config["icp_parameters"]["error_metric"]))
    reg.register()
    
    return reg 


def down_sample_both_clouds(scan_raw, model_raw, logger, k=1):
    """ Uniform downsample with voxel grid """
    mean_nn_model = pyreg.functions.get_mean_nn_distance(model_raw)
    mean_nn_scan = pyreg.functions.get_mean_nn_distance(scan_raw)
    
    size_model = k*mean_nn_model
    size_scan = k*mean_nn_scan
    model_raw.uniform_downsample(voxel_size=size_model)
    scan_raw.uniform_downsample(voxel_size=size_scan)
    
    model = copy.deepcopy(model_raw)
    scan = copy.deepcopy(scan_raw)
    logger.info("New scan size: %d", numpy.asarray(scan.points).shape[0])
    logger.info("New model size: %d", numpy.asarray(model.points).shape[0])
    
    return scan, model

def compute_fpfh(pcd, k=5, neighborhood=50):
    """ Compute normals and the fpfh """
    mean_nn = pyreg.functions.get_mean_nn_distance(pcd)
    
    # TODO Find good values that are indepent of sampling density!
    radius_normal = 4*mean_nn
    pcd.estimate_normals_open3d(neighborhood=neighborhood, search_radius=radius_normal)

    radius_feature = k*mean_nn
    pcd.extract_fpfh(radius=radius_feature, max_nn=neighborhood)
    
    return pcd


def main():
    """ 
    Register with the BCU approach implementation of Open3D
    """
    script_dir = os.getcwd()
    config = pyreg.functions.read_config("config.yml")
    logger = script_prequisites(script_dir, config)
    model_raw, scan_raw = read_data(script_dir, config, logger)    
    
    # Downsample
    scan, model = copy.deepcopy(scan_raw), copy.deepcopy(model_raw)
    
    # Features
    scan, model = compute_fpfh(scan), compute_fpfh(model)
    
    overlap_estimate = 1.0
    epsilon = config["convergence_criteria"]["relative_rmse"]
    max_iteration = config["convergence_criteria"]["max_iteration"]
    max_validation = config["convergence_criteria"]["max_validation"]
    # Do not use this as lightly/accurate as with other approaches
    epsilon = 1.5*pyreg.functions.get_mean_nn_distance(model)
    
    rigid_reg = Open3dFastBCURegistration(scan, model,
                                        ConvergenceCriteria(
                                            tolerance=epsilon
                                            )
                                        )
    reg_time, coarse_estimate = rigid_reg.register()
    
    scan.transform(coarse_estimate)
    pyreg.visualization.draw_registration(scan, model)
    
    ipdb.set_trace()
    
    result = refinement(scan, model, config, logger, trans_init=coarse_estimate, overlap=overlap_estimate)
    scan.transform(result.transformation)
    pyreg.visualization.draw_registration(scan_raw, model_raw)
    
    
    
if __name__ == "__main__":
    main()

