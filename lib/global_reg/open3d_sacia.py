#!/usr/bin/env python3

"""
Global coarse registration with FPFH features and advanced RANSAC outlier rejection 
"""

import os
import numpy 
import inspect
import ipdb
import open3d
import copy
import time
import pyreg.functions
from pyreg.point_cloud_class import *
from pyreg.local_reg.local_registration_base import ConvergenceCriteria
    

class Open3dSacIaRegistration(object):
    """ 
    RANSAC approach for coarse alingment based on FPFH
    
    This is the Open3D implementation of the SAC-IA algorithm from FPFH paper. It is supposedly very sophisticated, 
    but we had problems with this algorithm! To have more development freedom, we implemented this ourselves in pure Python
    with FPFH RANSAC...
    """
    # TODO Make all options callable
    def __init__(self, source, target, ConvergenceCriteria=None):
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
        
        
        if test_pcd_properties(source) == True and test_pcd_properties(target) == True:
            pass
        else:
            raise Exception("Point clouds do not have necessary preprocessing!")

        self.source = copy.deepcopy(source)
        self.target = copy.deepcopy(target)    
        self.source_open3d = convert_pcd_to_open3d(source)
        self.target_open3d = convert_pcd_to_open3d(target)
        self.source_fpfh = convert_fpfh_to_open3d(self.source.fpfh)
        self.target_fpfh = convert_fpfh_to_open3d(self.target.fpfh)
    
        if ConvergenceCriteria is None:
            mean_nn = source.get_mean_nn_distance()
            self.epsilon = 2*mean_nn
        else:
            self.epsilon = ConvergenceCriteria.tolerance
        self.max_iteration = 10000 if ConvergenceCriteria is None else ConvergenceCriteria.max_iteration
        self.max_validation = 500 if ConvergenceCriteria is None else ConvergenceCriteria.max_validation
        self.transformation = numpy.eye(4)
        self.time = 0.0
    
    
    def register(self):
        start = time.time()
        result = open3d.registration_ransac_based_on_feature_matching(
                    self.source_open3d, self.target_open3d, self.source_fpfh, self.target_fpfh,
                    self.epsilon,
                    open3d.TransformationEstimationPointToPoint(False), 4,
                    [
                    open3d.CorrespondenceCheckerBasedOnEdgeLength(0.95),
                    open3d.CorrespondenceCheckerBasedOnDistance(self.epsilon)
                    ],
                    open3d.RANSACConvergenceCriteria(self.max_iteration, self.max_validation)
                    )
        end = time.time()
        
        self.time = end-start
        self.transformation = result.transformation
        
        return 


class ConvergenceCriteria(object):      
    """ Subclass for convergence criteria """
    def __init__(self, max_iteration=None, tolerance=None, max_validation=None, min_inlier_size=None):
        self.max_iteration = 30 if max_iteration is None else max_iteration
        self.tolerance = 1e-6 if tolerance is None else tolerance
        self.max_validation = 500 if max_validation is None else max_validation
        self.min_inlier_size = 10 if min_inlier_size is None else min_inlier_size

