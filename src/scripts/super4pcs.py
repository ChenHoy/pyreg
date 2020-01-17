#!/usr/bin/env python3

"""
Super4PCs registration
"""

import os
import numpy 
import subprocess
import re
import ipdb
import copy
import time
from pyreg.global_reg.super4pcs import Super4PCSRegistration
from pyreg.local_reg.icpy.icp_registration import IcpRegistration
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


def main():
    """ 
    - Make a link to the executable in the subdirectory
    - Read in point clouds
    - Set parameters
    - Register with ./Super4PCs.exe
    - Get output from command line -> time it took
    - Get registration result from output_file
    - Turn this into a callable routine from somewhere else, so we can combine this with something else
    """
    script_dir = os.getcwd()
    config = pyreg.functions.read_config("config.yml")
    logger = script_prequisites(script_dir, config)
    model_raw, scan_raw = read_data(script_dir, config, logger)    
    pyreg.visualization.draw_registration(scan_raw, model_raw)
    
    model_filepath = script_dir+"/"+config["files"]["target"]
    scan_filepath = script_dir+"/"+config["files"]["source"]
    output_path = script_dir+"/"+config["files"]["output"]+"result.obj"
    
    overlap_estimate = 0.5
    epsilon = config["convergence_criteria"]["relative_rmse"] # TODO we need a meaningful value for this
    rigid_reg = Super4PCSRegistration(model_filepath, scan_filepath, output_path, 
                            #~ ConvergenceCriteria(tolerance=epsilon),
                            overlap=overlap_estimate
                            )
    reg_time, coarse_estimate = rigid_reg.register()
    
    scan_raw.transform(coarse_estimate)
    # Recenter, because the files may not be centered and we dont want to alter them
    # ALTERNATIVE: center files and then resave them, so we dont have a difference in translation
    # QUICK HACK
    scan_raw.center_around_origin()
    coarse_estimate[:-1 ,-1] = numpy.zeros(3)
    
    pyreg.visualization.draw_registration(scan_raw, model_raw)
    
    scan_raw.write_pcd(file_format="txt", filepath=script_dir+"/"+os.path.split(config["files"]["source"])[0]+"/OuterSheet_sampled_registered.txt") 
    model_raw.write_pcd(file_format="txt", filepath=script_dir+"/"+os.path.split(config["files"]["target"])[0]+"/extSheet_upsamples_1.ply_registered.txt") 
    
    ipdb.set_trace()
    
    result = refinement(scan_raw, model_raw, config, logger, trans_init=coarse_estimate, overlap=overlap_estimate)
    scan_raw.transform(result.transformation)
    pyreg.visualization.draw_registration(scan_raw, model_raw)
    
    remaining_distances = pyreg.functions.get_nn_distances(model_raw, scan_raw)
    pyreg.visualization.color_deviations(scan, remaining_distances)
    
    
    
if __name__ == "__main__":
    main()

