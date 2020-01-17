#!/usr/bin/env python3

"""
Compute matchings of a transformed set via the ICP
"""

import os
import sys
import time
import copy
import psutil
import numpy
from pyreg.local_reg.cpyd.cpd_rigid_registration import CpdRigidRegistration
from pyreg.local_reg.local_registration_base import ConvergenceCriteria
import pyreg.functions
import pyreg.visualization
from pyreg.point_cloud_class import *
import ipdb # whenever you use IPython together with mlab: ipython --gui=qt and then run script.py!


def script_prequisites(script_dir, config):
    """ Prepare the logger and directory """
    if config["files"]["save"] == True:
        timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
        output_path = config["files"]["output"]+"icp_"+timestr
        pyreg.functions.make_dir(output_path)
        pyreg.functions.write_config(config, output_path)
        log_file = output_path + "/icp.log"
        logger = pyreg.functions.create_timed_rotating_log(log_file)
        logger.info("New Run \n ##############################################")
        
        return output_path, logger
    
    else:
        timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
        log_file = "icp.log" # Make a logfile in current folder
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
    
    
def main():
    """ CPD registration on Stanford Bunny """
    script_dir = os.getcwd()
    config = pyreg.functions.read_config("config.yml")
    logger = script_prequisites(script_dir, config)
    model_raw, scan_raw = read_data(script_dir, config, logger)
    
    transformation = pyreg.functions.generate_transformation_zyx_euler_angles(30.0, 0.0, 0.0, 
                                                                              0.0, 0.0, 0.0)
    with pyreg.functions.show_complete_array():
        logger.info("Ground truth Transformation: \n %s", transformation)
    logger.info("Applying transformation...")
    scan_raw.transform(transformation)

    trans_init = numpy.identity(4)
    #~ pyreg.visualization.draw_registration(scan_raw, model_raw)
    
    model, scan = down_sample_both_clouds(scan_raw, model_raw, logger, k=2)
    #~ model, scan = copy.deepcopy(model_raw), copy.deepcopy(scan_raw)
    
    logger.info("Applying CPD...")
    start = time.time()
    reg = CpdRigidRegistration(copy.deepcopy(scan), copy.deepcopy(target), trans_init, config, logger, output_path,
                            ConvergenceCriteria(max_iteration=config["convergence_criteria"]["max_iteration"],
                                                tolerance=config["convergence_criteria"]["relative_rmse"]),
                                                )
    reg.register()
    end = time.time()
    time_cpd = end-start
    reg.times["total_time"] = time_cpd
    logger.info("Finished CPD, time: %f", time_cpd)
    logger.info("Number of iterations: %f", reg.iteration)
    
    with pyreg.functions.show_complete_array():
        logger.info("Estimated transformation: \n %s" , reg.transformation)
    logger.info("Remaining sigma: %f", reg.sigma2)
    
    scan.transform(reg.transformation)
    if config["visualization"] == True:
        logger.info("Visualizing estimate")
        pyreg.visualization.draw_registration(scan, target)
    else:
        pass
    
    if config["colorize_deviations"] == True:
        logger.info("Coloring point cloud with deviation color map")
        pyreg.functions.color_deviations(scan, distances)
    else:
        pass
    
    if config["plot_rmse"] == True:
        logger.info("Plotting rmse over iterations")
        pyreg.functions.plot_rmse(reg.true_err_history)
    else:
        pass
    
    
    if config["files"]["save"] == True:
        registration_results = {}
        registration_results["ground_truth"] = transformation
        registration_results["initial_transformation"] = reg._T_init
        registration_results["estimated_transformation"] = reg.transformation
        registration_results["final_iteration"] = reg.iteration
        registration_results["rmse_history"] = reg.true_err_history
        registration_results["times"] = reg.times
    
        pyreg.functions.save_pickle(registration_results, output_path+"/results")
    
    else:
        pass



if __name__ == "__main__":
    main()    

