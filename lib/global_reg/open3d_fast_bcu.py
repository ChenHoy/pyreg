#!/usr/bin/env python3

"""
Global coarse registration with FPFH features and advanced RANSAC outlier rejection 
"""

import os
import numpy 
import ipdb
import open3d
import copy
import time
import pyreg.functions
from pyreg.point_cloud_class import *
    

class Open3dFastBCURegistration(object):
    """ 
    New fast approach by Zhou et al, that is based on a Block Coordinate Descent or Block Coordinate Update (BCU) method:
        - Use an M-estimator in the regularization function of the global objective function
        - Alternate between optimizing line processes
        - and optimizing BCD for a fixed line process (Line search along the fixed Block aka Correspondence set)
        - The prior is based on the FPFH correspondence space we also used in RANSAC
    """
    # TODO make options callable
    def __init__(self, source, target, ConvergenceCriteria, division_factor=1.4, 
                 use_absolute_scale=0, decrease_mu=229, iteration_number=64, tuple_scale=0.95, max_tuple_count=1000):
        def test_pcd_properties(pcd):
            """ Both point clouds need to have features  """   
            if pcd.has_points() == False or pcd.has_fpfh() == False:
                return False
            else:
                return True
    
        def convert_pcd_to_open3d(pcd):
            """ Convert from PointCloud() to open3d.Vector3dVector """
            if isinstance(pcd, PointCloud) == True:
                new_pcd = open3d.PointCloud()
                new_pcd.points = open3d.Vector3dVector(pcd.points)
                if pcd.has_normals() == True:
                    new_pcd.normals = open3d.Vector3dVector(pcd.normals)
                else:
                    pass
                
                return new_pcd
                
            elif isinstance(pcd, numpy.ndarray) == True:
                new_pcd = open3d.PointCloud()
                new_pcd.points = open3d.Vector3dVector(pcd)
                
                return new_pcd
                
            elif isinstance(pcd, open3d.PointCloud) == True:
                return pcd
        
        def convert_fpfh_to_open3d(fpfh):
            """ Convert back to Open3d format of FPFH """
            open3d_fpfh = open3d.Feature()
            open3d_fpfh.data = fpfh.T # They have a transposed format
        
            return open3d_fpfh
        
        self.distance_threshold = ConvergenceCriteria.tolerance
        if test_pcd_properties(source) == True and test_pcd_properties(target) == True:
            pass
        else:
            raise Exception("Point clouds do not have necessary preprocessing!")

        self.source = copy.deepcopy(source)
        N = self.source.points.shape[0]
        self.target = copy.deepcopy(target)    
        self.source_open3d = convert_pcd_to_open3d(source)
        self.target_open3d = convert_pcd_to_open3d(target)
        self.source_fpfh = convert_fpfh_to_open3d(self.source.fpfh)
        self.target_fpfh = convert_fpfh_to_open3d(self.target.fpfh)
        self.transformation = numpy.eye(4)
        self.correspondence_set = numpy.zeros((N, 2))
        self.inlier_rmse = 10.0
        self.time = 0.0
        self.division_factor = division_factor
        self.use_absolute_scale = use_absolute_scale
        self.decrease_mu = decrease_mu
        self.iteration_number = iteration_number
        self.tuple_scale = tuple_scale
        self.max_tuple_count = max_tuple_count
        
    def register(self):
        start = time.time()
        result = open3d.registration_fast_based_on_feature_matching(
                    self.source_open3d, self.target_open3d, self.source_fpfh, self.target_fpfh,
                    open3d.FastGlobalRegistrationOption(
                        maximum_correspondence_distance = self.distance_threshold,
                        division_factor=self.division_factor,
                        use_absolute_scale=0,
                        decrease_mu=self.decrease_mu,
                        iteration_number=self.iteration_number,
                        tuple_scale=self.tuple_scale,
                        maximum_tuple_count=self.max_tuple_count
                        )
                    )
        
        end = time.time()
        self.time = end-start
        self.transformation = numpy.asarray(result.transformation)
        self.correspondence_set = numpy.asarray(result.correspondence_set)
        self.inlier_rmse = numpy.asarray(result.inlier_rmse)
        
        return 

class ConvergenceCriteria(object):      
    """ Subclass for convergence criteria """
    def __init__(self, tolerance=None, max_iteration=64, decrease_mu=229, 
                 division_factor=1.4, use_absolute_scale=0, tuple_scale=0.95):
        
        self.max_iteration = max_iteration 
        self.tolerance = 1e-6 if tolerance is None else tolerance
        self.decrease_mu = decrease_mu
        self.division_factor = division_factor
        self.use_absolute_scale = use_absolute_scale
        self.tuple_scale = tuple_scale
