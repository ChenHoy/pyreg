#!/usr/bin/env python3

"""
Compute matchings of a transformed set via the ICP
"""

import os
import numpy 
import copy
import time
import open3d
import matplotlib.pyplot
from pyreg.icpy.icp_registration import IcpRegistration
from pyreg.cpyd.cpd_registration import CpdRigidRegistration
from pyreg.registration_base import ConvergenceCriteria
from pyreg.registration_base import true_mean_squared_error
import pyreg.functions
#~ import pyreg.visualization
from pyreg.point_cloud_class import *
import ipdb # whenever you use IPython together with mlab: ipython --gui=qt and then run script.py!


def downsample(pcd, voxel_size):
    pcd_open3d = open3d.PointCloud()
    pcd_open3d.points = open3d.Vector3dVector(pcd.points)
    
    pcd_open3d = open3d.voxel_down_sample(pcd_open3d, voxel_size=voxel_size)
    pcd = PointCloud(points=copy.deepcopy(numpy.asarray(pcd_open3d.points)))
    
    return pcd

    
def post_processing(config, logger, scan, target, result, iterations, sizes):
    def plot_iteration_count(sizes, iterations):
        fig = matplotlib.pyplot.figure()
        ax = matplotlib.pyplot.axes()
        matplotlib.pyplot.ylabel("iterations", fontsize=15)
        matplotlib.pyplot.xlabel("resolution: number of used points", fontsize=15)
        x = numpy.arange(len(sizes))
        matplotlib.pyplot.bar(x, iterations) 
        
        labels=[]
        for element in sizes:
            labels.append(str(element))
            
        matplotlib.pyplot.xticks(x, labels) 
        matplotlib.pyplot.show()
        
        return
    
    scan.transform(result.transformation)
    if config["visualization"] == True:
        logger.info("Visualizing estimate")
        #~ pyreg.visualization.draw(scan, target)
    else:
        pass
    
    plot_iteration_count(sizes, iterations)
    
    logger.info("Calculating remaining deviations...")
    distances = numpy.sqrt(((scan.points - target.points)**2).sum(axis=1))
    logger.info("Maximum deviation: %.17f",  numpy.asarray(distances).max())
    
    if config["colorize_deviations"] == True:
        logger.info("Coloring point cloud with deviation color map...")
        pyreg.functions.color_deviations(scan, distances)
    else:
        pass
    
    if config["plot_deviations"] == True: 
        logger.info("Plotting histogram of remaining deviations...")
        distances = numpy.asarray(distances)
        pyreg.functions.distances_histogram(distances)
    else:
        pass 
    

def prequisites():
    """ 
    - Config
    - Log file
    """
    script_dir = os.getcwd()
    config = pyreg.functions.read_config("config.yml")
    timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
    log_file = "ICP.log" # Make a logfile in current folder
    logger = pyreg.functions.create_timed_rotating_log(log_file) 
    logger.info("New Run \n ##############################################")
    
    return script_dir, config, logger


def create_problem(script_dir, config, logger):
    """ Create a registration problem from one data set """
    def read_pcds(script_dir, config, logger):
        logger.info("Reading input point clouds...")
        target = PointCloud()
        target.read_pcd(script_dir+"/"+config["files"]["target"])
        target.center_around_origin()
        scan = copy.deepcopy(target)
        logger.info("Source point cloud size: %d", scan.points.shape[0])
        logger.info("Target point cloud size: %d", target.points.shape[0])
        
        return scan, target
    
    scan, target = read_pcds(script_dir, config, logger)
    
    transformation = pyreg.functions.generate_transformation_zyx_euler_angles(50.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    with pyreg.functions.show_complete_array():
        logger.info("Ground truth Transformation: \n %s", transformation)
    scan.transform(transformation)
    
    if config["visualization"] == True:
        logger.info("Visualizing registration problem")
        #~ pyreg.visualization.draw(scan, target)
    else:
        pass
    
    return scan, target


def registration(config, logger, scan, target):
    """ Registration with point to point metric on Stanford Bunny """
    def estimate_normals(scan, target, config):
        mean_nn = pyreg.functions.get_mean_nn_distance(scan)
        scan.estimate_normals_open3d(neighborhood=config["icp_parameters"]["estimate_normals"]["max_nn"], 
                                     search_radius=config["icp_parameters"]["estimate_normals"]["search_radius"]*mean_nn)
        mean_nn = pyreg.functions.get_mean_nn_distance(target)
        target.estimate_normals_open3d(neighborhood=config["icp_parameters"]["estimate_normals"]["max_nn"], 
                                     search_radius=config["icp_parameters"]["estimate_normals"]["search_radius"]*mean_nn)
                                     
        return
    
    def CPD(scan, target, trans_init, config, logger, w):
        logger.info("Applying CPD...")    
        start = time.time()
        reg = CpdRegistration(copy.deepcopy(scan), copy.deepcopy(target), trans_init, config, logger, 
                                ConvergenceCriteria(max_iteration=config["convergence_criteria"]["max_iteration"],
                                                    tolerance=config["convergence_criteria"]["relative_rmse"],
                                                    w=w
                                                    )
        reg.register()    
        end = time.time()
        reg_time = end-start
        reg.times["total_time"] = reg_time
        
        return reg
    

    def ICP(scan, target, trans_init, config, logger):
        logger.info("Applying ICP...")    
        start = time.time()
        reg = IcpRegistration(copy.deepcopy(scan), copy.deepcopy(target), trans_init, config, logger, 
                                ConvergenceCriteria(max_iteration=config["convergence_criteria"]["max_iteration"],
                                                    tolerance=config["convergence_criteria"]["relative_rmse"],
                                                    error_metric=config["icp_parameters"]["error_metric"]))
        reg.register()    
        end = time.time()
        reg_time = end-start
        reg.times["total_time"] = reg_time
        
        return reg
    
    num_scales = 7
    min_size = 500 
    mean_nn_scan = pyreg.functions.get_mean_nn_distance(scan)    
    mean_nn_target = pyreg.functions.get_mean_nn_distance(target)
    
    # Init
    trans_init = numpy.identity(4)
    iterations = []
    scales= []
    sizes = []
    estimates = []
    current_size = scan.points.shape[0]
    previous_size = current_size
    
    start = time.time()
    for i in range(num_scales+1):
        resolution = num_scales-i
        voxel_size_scan = resolution*mean_nn_scan
        voxel_size_target = resolution*mean_nn_target
        if resolution != 0:
            scan_down = downsample(copy.deepcopy(scan), voxel_size_scan)
            target_down = downsample(copy.deepcopy(target), voxel_size_target)
        else:
            scan_down = copy.deepcopy(scan)
            target_down = copy.deepcopy(target)
        
        current_size = scan_down.points.shape[0]
        delta_size = abs(current_size-previous_size)
        if scan_down.points.shape[0] < min_size or target_down.points.shape[0] < min_size:
            continue
        else:
            if delta_size < 2000:
                continue
            else:
                previous_size = current_size
                if config["icp_parameters"]["error_metric"] == "point_to_plane":
                    estimate_normals(scan_down, target_down, config)
                else:
                    pass
                if config["algorithm"] == "icp":
                    result = ICP(scan_down, target_down, trans_init, config, logger)
                elif config["algorithm"] == "cpd":
                    result = CPD(scan_down, target_down, trans_init, config, logger, w=config["cpd_parameters"]["w"])
                else:
                    raise Exception("Algorithm not supported!")
                
                logger.info("Finished registration, time: %f \n iterations: %d", result.times["total_time"], result.iteration)
                with pyreg.functions.show_complete_array():
                    logger.info("Estimated transformation: \n %s" , result.transformation)
        
                logger.info("Last rmse is: %.17f", result.inlier_rmse)
        
                trans_init = result.transformation @ trans_init
                
                iterations.append(result.iteration)
                scales.append(resolution)
                sizes.append(scan_down.points.shape[0])
                estimates.append(trans_init)

                if config["algorithm"] == "icp":
                    if result.inlier_rmse < 1e-6:
                        break
                    else:
                        pass
                elif config["algorithm"] == "cpd":
                    if result.current_err < 1e-6:
                        break
                    else:
                        pass
                else:
                    raise Exception("No algorithm")

    end = time.time()
    reg_time = end - start
    
    return result, iterations, sizes, reg_time, estimates

    
if __name__ == "__main__":
    script_dir, config, logger = prequisites()
    scan, target = create_problem(script_dir, config, logger)    
    result, iterations, sizes, reg_time, estimates = registration(config, logger, scan, target)
    
    print(reg_time)
    ipdb.set_trace()
    
    #~ post_processing(config, logger, scan, target, result, iterations, sizes)
