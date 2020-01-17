#!/usr/bin/env python3

"""
RANSAC registration after Key point extraction with FPFH
"""

import os
import numpy 
import bottleneck
from functools import reduce
import ipdb
import copy
import time
import matplotlib.pyplot
from pyreg.local_reg.icpy.icp_registration import IcpRegistration
from pyreg.local_reg.local_registration_base import ConvergenceCriteria
from pyreg.local_reg.icpy.icp_registration import Selection
from pyreg.local_reg.icpy.icp_registration import Rejection
from pyreg.local_reg.icpy.icp_registration import Weighting
from pyreg.local_reg.icpy.icp_registration import LevenbergMarquardt
from pyreg.global_reg.fpfh_ransac import ConvergenceCriteria as RansacTerminationCriteria
from pyreg.global_reg.fpfh_ransac import FpfhRANSAC
import scipy.spatial.distance
import scipy.stats
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
    
    
def compute_features(pcd, k=5, neighborhood=50):
    """ Compute normals and the fpfh """
    mean_nn = pyreg.functions.get_mean_nn_distance(pcd)
    
    # TODO Find good values that are indepent of sampling density!
    if pcd.has_normals() == False:
        radius_normal = 4*mean_nn
        pcd.estimate_normals_open3d(neighborhood=neighborhood, search_radius=radius_normal)
    else:
        pass

    radius_feature = k*mean_nn
    pcd.extract_fpfh(radius=radius_feature, max_nn=neighborhood)
    
    return pcd


def plot_mu_fpfh_histogram(pcd):
    """ 
    We plot the mean 33-dimensional feature histogram 
    
    Salient points can be found from this histogram, by finding a quartile of highest distance histograms
    """
    mu_fpfh = numpy.mean(pcd.fpfh, axis=0)
    matplotlib.pyplot.bar(range(0, mu_fpfh.shape[0]), mu_fpfh)
    matplotlib.pyplot.show()
    
    return


def extract_key_points(pcd, feature="fpfh", points="variable", quartile=0.6, num_points=100, dist_metric="cityblock"):
    """ Extract points with a high distance to the mu-histogramm. These are more likely to be salient key points """
    def compute_distances_to_mu_histogram(pcd , metric="cityblock"):
        mu_fpfh = numpy.mean(pcd.fpfh, axis=0).reshape(1, pcd.fpfh.shape[1])
        if metric == "cityblock":
            dist = scipy.spatial.distance.cdist(pcd.fpfh, mu_fpfh, metric="cityblock")
        elif metric == "euclidean":
            dist = scipy.spatial.distance.cdist(pcd.fpfh, mu_fpfh, metric="euclidean")
        elif metric == "kl":            
            dist = numpy.zeros(pcd.fpfh.shape[0])
            for i in range(pcd.fpfh.shape[0]):
                dist[i] = scipy.stats.entropy(pcd.fpfh[i], mu_fpfh.flatten(), base=None) # Kulback-Leibler divergence
        else:
            raise Exception("Specified metric not in use!")
        
        return dist
    
    if feature == "fpfh":
        fpfh_distances = compute_distances_to_mu_histogram(pcd, metric=dist_metric)
        dist_mean = numpy.mean(fpfh_distances)
        dist_std = numpy.std(fpfh_distances)
        
        if points == "variable":
            confid_interval = scipy.stats.norm.interval(quartile, loc=dist_mean, scale=dist_std)
            key_points_idx = numpy.where((fpfh_distances < confid_interval[0]) | (fpfh_distances > confid_interval[1]))[0]
        elif points == "fixed":
            # This is the fastes partial sort method! The array is sorted so that the n first elements are the largest (-a, n[:n])
            key_points_idx = bottleneck.argpartition(-fpfh_distances, num_points)[:num_points]
        else:
            raise Exception("Choose (fixed) or (variable) for points! ")
        
        pcd.key_points = [pcd.points[key_points_idx], key_points_idx]
    else:
        raise Exception("Other features are not supported yet!")
    
    return


def find_persistent_key_points(pcd, feature="fpfh", dist_metric="cityblock"):
    """ 
    Persistenc analysis: 
        - Compute features at different radii
        - For each radii, exract key points via their feature distance to the mean feature value
        - Remember key points for each radius
        - Key points that appear in all sets are persistent and extracted
    """ 
    #~ scales = [15, 10, 5, 2] # Is this good??
    scales = [10, 5, 2]
    key_sets = []
    for k in scales:
        pcd = compute_features(pcd, k=k, neighborhood=30)
        extract_key_points(pcd, points="variable", quartile=0.6, dist_metric=dist_metric)
        key_sets.append(pcd.key_points[1])
    
    persistent_key_points = reduce(numpy.intersect1d, (key_sets[0], key_sets[1], key_sets[2])) # TODO Make this flexible
    pcd.key_points = [pcd.points[persistent_key_points], persistent_key_points]
    
    return

    
def refinement(scan, target, config, trans_init=numpy.eye(4)):
    """ Refine the RANSAC result with ICP """
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
    
    reg = IcpRegistration(copy.deepcopy(scan), copy.deepcopy(target), trans_init, 
                            ConvergenceCriteria(max_iteration=config["convergence_criteria"]["max_iteration"],
                                                tolerance=config["convergence_criteria"]["relative_rmse"]
                                                ),
                            Selection=Selection(kind=config["icp_parameters"]["selection"]["kind"],
                                                samples=config["icp_parameters"]["selection"]["samples"]
                                                ),
                            Weighting=Weighting(
                                                kind=config["icp_parameters"]["weighting"]["kind"]
                                                ),
                            Rejection=Rejection(heuristics=config["icp_parameters"]["rejection"]["heuristics"], 
                                                max_distance=config["icp_parameters"]["rejection"]["max_distance"],
                                                trim_percentage=config["icp_parameters"]["rejection"]["trim"],
                                                iqr_factor=config["icp_parameters"]["rejection"]["iqr_factor"]
                                                ),
                            Error_metric=config["icp_parameters"]["error_metric"],
                            LM=LevenbergMarquardt(lambda_init=config["icp_parameters"]["lm_params"]["lambda"],
                                                  multiplier=config["icp_parameters"]["lm_params"]["multiplier"],
                                                  m_estimator=config["icp_parameters"]["lm_params"]["m_estimator"]
                                                  )
                            )
    reg.register()
    
    return reg
    

def main():
    """ Test our own registration algorithm """    
    script_dir = os.getcwd()
    config = pyreg.functions.read_config("config.yml")
    logger = script_prequisites(script_dir, config)
    
    model_raw, scan_raw = read_data(script_dir, config, logger)
    
    transformation = pyreg.functions.generate_transformation_zyx_euler_angles(180.0, 120.0, -60.0, 
                                                                              0.0, 0.0, 0.0)
    with pyreg.functions.show_complete_array():
        logger.info("Ground truth Transformation: \n %s", transformation)
    logger.info("Applying transformation...")
    scan_raw.transform(transformation)
    
    #~ pyreg.visualization.draw_registration(scan_raw, model_raw)    
    
    # Registration
    model, scan = copy.deepcopy(model_raw), copy.deepcopy(scan_raw) 
    #~ model, scan = down_sample_both_clouds(copy.deepcopy(scan_raw), copy.deepcopy(model_raw), logger, k=4)
    mean_nn_model = pyreg.functions.get_mean_nn_distance(model)
    mean_nn_scan = pyreg.functions.get_mean_nn_distance(scan)
    
    scan, model = compute_features(scan), compute_features(model)
    
    coarse_reg = FpfhRANSAC(scan, model, 
                           RansacTerminationCriteria(
                                        max_iteration=config["ransac_parameters"]["max_iteration"],
                                        max_validation=config["ransac_parameters"]["max_validation"],
                                        tolerance=2*mean_nn_model,
                                        min_inlier_size=config["ransac_parameters"]["inlier_size"]
                                    ), 
                            size_subset=config["ransac_parameters"]["subset_size"]
                            )
    coarse_reg.register()
    
    #~ pyreg.visualization.draw_correspondences(scan, model, coarse_reg.found_inliers)
    scan.transform(coarse_reg.transformation)
    #~ pyreg.visualization.draw_registration(scan, model)
    
    fine_reg = refinement(scan, model, config)
    scan_raw.transform(fine_reg.transformation)
    pyreg.visualization.draw_registration(scan, model)
    

    
if __name__ == "__main__":
    main()    

