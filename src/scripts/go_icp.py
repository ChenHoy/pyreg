#!/usr/bin/env python3

"""
Go-ICP registration
"""

import os
import numpy 
import ipdb
import copy
import time
from pyreg.global_reg.go_icp import GoICPRegistration 
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
    
    logger.info("Source point cloud size: %d", source_raw.points.shape[0])
    logger.info("Target point cloud size: %d", target_raw.points.shape[0])
    
    return target_raw, source_raw
  

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

    

def main():
    """ 
    - Read in point clouds
    - Downsample if needed
    - Register with Go-ICP
    - Compute results
    """
    script_dir = os.getcwd()
    config = pyreg.functions.read_config("config.yml")
    logger = script_prequisites(script_dir, config)
    model_raw, scan_raw = read_data(script_dir, config, logger)    
    
    scan, model = down_sample_both_clouds(scan_raw, model_raw, logger, k=3)

    scan_filepath = script_dir+"/"+config["files"]["source"]
    model_filepath = script_dir+"/"+config["files"]["target"]
    output_path = script_dir+"/"+config["files"]["output"]+"result.txt"
    config_path = script_dir+"/"+"go_icp_config.txt"
    
    overlap_estimate = 0.6
    epsilon = config["convergence_criteria"]["relative_rmse"]
    
    rigid_reg = GoICPRegistration(scan, model, model_filepath, scan_filepath, config_path, output_path,
                                  epsilon, overlap=overlap_estimate)
    reg_time, transformation_estimate = rigid_reg.register()
    
    scan.transform(transformation_estimate)
    pyreg.visualization.draw_registration(scan, model)
    
    ipdb.set_trace()
    
        
if __name__ == "__main__":
    main()

