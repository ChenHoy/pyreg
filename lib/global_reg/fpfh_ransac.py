import numpy 
import ipdb
import copy
import time
import math
from pyreg.registration_base import RegistrationBase
import pyreg.visualization
from pyreg.point_cloud_class import *


class FpfhRANSAC(RegistrationBase):
    """
    greedy RANSAC approach for filtering out bad FPFH correspondences 
    """
    def __init__(self, source, target, ConvergenceCriteria=None, size_subset=1, trim=None, feature_metric="cityblock"):
        RegistrationBase.__init__(self, source, target)
        
        if self.SCAN.has_fpfh() == True and self.TARGET.has_fpfh() == True:
            pass
        else:
            raise Exception("Point clouds do not have necessary preprocessing! Compute FPFH features!")
        
        if ConvergenceCriteria is None:
            scan_density = source.get_mean_nn_distance()
            self.epsilon = 1.5*scan_density
        else:
            self.epsilon = ConvergenceCriteria.tolerance
        
        self.max_iteration = 1000 if ConvergenceCriteria is None else ConvergenceCriteria.max_iteration
        self.min_inlier_size = int(source.points.shape[0]*0.4) if ConvergenceCriteria is None else ConvergenceCriteria.min_inlier_size
        self.size_inliers = 0
        self.size_subset = 1 if size_subset is None else size_subset
        self.iteration = 0.0
        self.trim = trim
        self.feature_metric = feature_metric
        self.residuals = 10*numpy.ones(self.N)
        self.fitness = 10.0 # Something big
    
    
    def _transform(self, points, transformation):
        R = transformation[:points.shape[1], :points.shape[1]]
        t = transformation[:-1, -1]
        t_points = points @ R.T + numpy.tile(t.T, (points.shape[0], 1))
    
        return t_points
    
    
    def _find_feature_correspondences(self, scan, target, feature_metric="cityblock", trim=None):
        """ Search for nearest neighbor in the feature space """
        def trim_correspondences(corrrespondences, nn_distances, trim_percentage):
            """ Discard correspondences with a large feature distance """
            num = int(math.floor((1-trim_percentage)*corrrespondences.shape[0]))
            distances_sorted_idx = nn_distances.argsort(axis=0) 
            best_correspondences_idx = distances_sorted_idx[:num].flatten()
            trimmed_correspondences = corrrespondences[best_correspondences_idx]
            
            return trimmed_correspondences
        
        
        # TODO This improves robustness a lot and can reduce computation times
        def bidirectionality_test(scan, target, correspondences, feature_metric):
            """
            We already computed the nearets neighbors of scan from target. Now we will check wether the same
            neighbors are found when reversing the search. If yes, then keep the correspondence, else discard it.
            """
            dummy, reversed_nn_indices = RegistrationBase.find_nn(target.fpfh, scan.fpfh, dist_metric=feature_metric)
            reversed_correspondences = numpy.zeros(reversed_nn_indices.shape[0])
            reversed_correspondences[:, 1] = reversed_nn_indices.flatten() # MODEL/TARGET points
            reversed_correspondences[:, 0] = numpy.arange(size).flatten() # SCAN
            reversed_correspondences = numpy.int_(feature_correspondences)
    
        
        size = scan.fpfh.shape[0]
        feature_correspondences = numpy.zeros((size, 2)) # Init
        
        # Search through feature space
        nn_distances, nn_indices = RegistrationBase.find_nn(scan.fpfh, target.fpfh, dist_metric=feature_metric) 
        feature_correspondences[:, 1] = nn_indices.flatten() # MODEL/TARGET points
        feature_correspondences[:, 0] = numpy.arange(size).flatten() # SCAN
        feature_correspondences = numpy.int_(feature_correspondences)
        
        if trim != None:    
            feature_correspondences = trim_correspondences(feature_correspondences, nn_distances, trim)
        else:
            pass
    
        return feature_correspondences
    
    
    def _build_model(self, scan, target, feature_correspondences):
        maybe_inlier_correspondences = RegistrationBase.random_draw(feature_correspondences, size_sample=3, return_idx=False)
        base_q = scan.points[maybe_inlier_correspondences[:, 0].flatten()]
        base_p = target.points[maybe_inlier_correspondences[:, 1].flatten()]
        R, t = RegistrationBase.Horns_method(base_q, base_p) # Compute transformation from Q to P
        model_guess = numpy.eye(self.D+1)
        model_guess[:self.D, :self.D], model_guess[:self.D, -1] = R, t
        
        return model_guess, maybe_inlier_correspondences
    
    
    def _rebuild_model(self, scan, target, inlier_correspondences):
        """ 
        After finding an appropiate amount of inliers, we rebuild the transformation guess
        """
        base_q = scan.points[inlier_correspondences[:, 0].flatten()]
        base_p = target.points[inlier_correspondences[:, 1].flatten()]
        R, t = RegistrationBase.Horns_method(base_q, base_p) # Compute transformation from Q to P
        model_guess = numpy.eye(self.D+1)
        model_guess[:self.D, :self.D], model_guess[:self.D, -1] = R, t
        
        base_q_projected = self._transform(base_q, model_guess)
        residuals = numpy.linalg.norm(base_q_projected - base_p, axis=1)
        fitness = numpy.sqrt(numpy.mean((residuals**2)))
        
        return model_guess, fitness
    
    
    def _find_more_inliers(self, scan, target, model_guess, epsilon):
        """ Add more inliers for the model guess when their error is withing epsilon band """
        def _check_epsilon(nn_distances, nn_indices, num_matches):
            inlier_idx = numpy.where(nn_distances < epsilon)[0]
            target_match_idx = nn_indices[inlier_idx].flatten() 
            scan_match_idx = numpy.arange(num_matches)[inlier_idx].flatten() 
        
            return scan_match_idx, target_match_idx
        
        scan.transform(model_guess)
        nn_distances, nn_indices = RegistrationBase.find_nn(scan.points, target.points)
        num_matches = scan.points.shape[0]
        scan_match_idx, target_match_idx = _check_epsilon(nn_distances, nn_indices, num_matches)
        
        inlier_correspondences = numpy.zeros((target_match_idx.shape[0], 2)) # Overwritten each iteration
        inlier_correspondences[:, 1] = target_match_idx.flatten()  # Model
        inlier_correspondences[:, 0] = scan_match_idx.flatten() # Scan
        inlier_correspondences = numpy.int_(inlier_correspondences)
        
        return inlier_correspondences
    
    
    def _t_dd_test(self, scan, target, model_guess, ran_correspondences, feature_correspondences, size_subset, epsilon):
        """
        T_d,d test: Test hypothesis against small subset d from Q. If it is correct, then test the hypothesis completely!
        """
        def _check_epsilon(nn_distances, nn_indices, epsilon):
            """ Check if all points of the subset fall within epsilon upon projection """
            inlier_idx = numpy.where(nn_distances < epsilon)[0]
            num_inliers = inlier_idx.shape[0]
            size_subset = nn_distances.shape[0]
            if num_inliers == size_subset:
                return True
            else:
                return False
        
        more_inliers = numpy.setdiff1d(feature_correspondences[:, 0], ran_correspondences[:, 0]) # Exclude the base from set
        scan_without_base = copy.deepcopy(scan)
        scan_without_base.points = scan.points[more_inliers.flatten()]
        ran_subset, ran_subset_idx = RegistrationBase.random_draw(scan_without_base.points, size_sample=size_subset, return_idx=True)
        projected_subset = self._transform(ran_subset, model_guess)
        nn_distances, nn_indices = RegistrationBase.find_nn(projected_subset, target.points)
        test_result = _check_epsilon(nn_distances, nn_indices, epsilon)
    
        return test_result
    
    
    def register(self):
        start = time.time()
        
        self.feature_correspondences = self._find_feature_correspondences(self.SCAN, self.TARGET, feature_metric=self.feature_metric,
                                                                          trim=self.trim)
        for i in range(self.max_iteration):    
            model_guess, maybe_inlier_correspondences = self._build_model(self.SCAN, self.TARGET, self.feature_correspondences)
            
            test_result = self._t_dd_test(self.SCAN, self.TARGET, model_guess, maybe_inlier_correspondences, self.feature_correspondences, 
                                    self.size_subset, epsilon=2*self.epsilon)
            if test_result == True:
                new_inlier_set = self._find_more_inliers(copy.deepcopy(self.SCAN), copy.deepcopy(self.TARGET), 
                                                    model_guess, epsilon=self.epsilon)
                self.found_inliers = numpy.append(maybe_inlier_correspondences, new_inlier_set, axis=0)
                self.size_inliers = self.found_inliers.shape[0]
                
                if self.size_inliers > self.min_inlier_size:
                    better_guess, current_fitness = self._rebuild_model(copy.deepcopy(self.SCAN), copy.deepcopy(self.TARGET), 
                                                                    self.found_inliers)
                    if current_fitness < self.fitness:
                        self.fitness = current_fitness
                        self.transformation = better_guess
                    else:
                        self.iteration += 1
                        continue
                    
                    if self.fitness < self.epsilon:
                        if self.standardize == True:            
                            t = self.transformation[:-1, -1] 
                            R = self.transformation[:self.D, :self.D]
                            self.transformation[:-1, -1] = self.scale*t + self.target_mean.T - R @ self.scan_mean.T
                        
                        end = time.time()
                        self.time = end-start
                        return
                    else:
                        self.iteration += 1
                        continue
                else:
                    self.iteration += 1
                    continue
            else:
                self.iteration += 1
                continue
    
        # If we have found no guess at all that has the required fitness
        end = time.time()
        self.time = end-start
        raise Warning("No good guess found, all iterations exhausted!")
        
        return 
    

class ConvergenceCriteria(object):      
    """ Subclass for convergence criteria """
    def __init__(self, max_iteration=None, tolerance=None, max_validation=None, min_inlier_size=None):
        self.max_iteration = 30 if max_iteration is None else max_iteration
        self.tolerance = 1e-6 if tolerance is None else tolerance
        self.min_inlier_size = 100 if min_inlier_size is None else min_inlier_size
