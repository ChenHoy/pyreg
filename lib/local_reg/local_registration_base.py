import numpy
import ipdb
from pyreg.point_cloud_class import *
from pyreg.registration_base import RegistrationBase
import copy


class LocalRigidRegistration(RegistrationBase):    
    """  
    Local base class
  
    Attributes:
        iteration (int)         : Iteration count 
        maxIterations (int)     : Maximum number of iterations
        inlier_rmse (float)     : Root Mean Square error
        tolerance (float)       : Stopping tolerance
        err_history (ndarray)   : Save the error over iterations for plot
        time (dict)             : Save times for individual steps, i.e. E-step, M-estpe and Iteration
    """
    def __init__(self, SCAN, TARGET, T_init, ConvergenceCriteria=None, standardize=False):
        RegistrationBase.__init__(self, SCAN, TARGET, T_init, standardize=standardize)
        
        self.ConvergenceCriteria    = None if ConvergenceCriteria is None else ConvergenceCriteria
        
        self.iteration              = 0
        self.max_iteration          = 50 if ConvergenceCriteria is None else ConvergenceCriteria.max_iteration
        self.tolerance              = 1e-6 if ConvergenceCriteria is None else ConvergenceCriteria.tolerance 
        self.times                  = {}
        self.times["estimation"]    = []
        self.times["maximization"]  = []
        self.times["iteration"]     = []
    
        self.current_err            = 1.0
        self.err_delta              = 1.0
        self.err_history            = []
    
    
    def Horns_method(X, Y, weighting=False, weights=None):
        R, t = RegistrationBase.Horns_method(X, Y, weighting=weighting, weights=weights)
        return R, t
    
    
    def find_nn(source, target, dist_metric="euclidean"):
        nn_distances, nn_indices = RegistrationBase.find_nn(source, target, dist_metric=dist_metric)
        return nn_distances, nn_indices
    
    
    def destandardize_data(self):
        RegistrationBase.destandardize_data(self)
        
        return
    
    def random_draw(source, size_sample, return_idx=False):
        if return_idx == True:
            ran_sample, ran_idx = RegistrationBase.random_draw(source, size_sample, return_idx=return_idx)
            return ran_sample, ran_idx
        else:
            ran_sample = RegistrationBase.random_draw(source, size_sample, return_idx=return_idx)
            return ran_sample


    def m_estimator(residuals, kernel="huber"):
        m_residuals = RegistrationBase.m_estimator(residuals, kernel=kernel)
        return m_residuals
    

class ConvergenceCriteria(object):      
    """ Subclass for convergence criteria """
    def __init__(self, max_iteration=None, error_metric=None, tolerance=None, max_validation=None):
        self.max_iteration = 30 if max_iteration is None else max_iteration
        self.tolerance = 1e-6 if tolerance is None else tolerance
