import numpy
import ipdb
from pyreg.point_cloud_class import *
from sklearn import preprocessing
import copy


class RegistrationBase(object):    
    """  
    Rigid registration base class to inherit from.
  
    Attributes:
        SCAN (ndarray)              : model point set (input)
        TARGET (ndarray)            : measured point set (manually transformed)
        T_SCAN (ndarray)            : Transformed point set
        N (int)                     : Number of points in SCAN
        m (int)                     : Number of points in TARGET
        D (int)                     : Dimension of the Data (=3)
        transformation (ndarray)    : total transformation
    
        iteration (int)         : Iteration count 
        maxIterations (int)     : Maximum number of iterations
        inlier_rmse (float)     : Root Mean Square error
        tolerance (float)       : Stopping tolerance
        err_history (ndarray)   : Save the error over iterations for plot
    """
    def __init__(self, SCAN, TARGET, T_init=None, standardize=False):
        """ 
        Initializes the object(set_SCAN, set_TARGET) and check exceptions 
        """
        if isinstance(SCAN, PointCloud) ==True:
            pass
        elif isinstance(SCAN, numpy.ndarray) ==True:
            SCAN_new = PointCloud()
            SCAN_new.points = SCAN
            SCAN = copy.deepcopy(SCAN_new)
        else:
            raise TypeError("Input scan has to be a PointCloud object or an ndarray. Transform data into correct format!")
        
        if isinstance(TARGET, PointCloud) ==True:
            pass
        elif isinstance(TARGET, numpy.ndarray) ==True:
            TARGET_new = PointCloud()
            TARGET_new.points = TARGET
            TARGET = copy.deepcopy(TARGET_new)
        else:
            raise TypeError("Input target has to be a PointCloud object or an ndarray. Transform data into correct format!")
        
        self.SCAN                   = SCAN
        self.TARGET                 = TARGET
        
        (self.N, self.D)            = self.SCAN.points.shape
        (self.M, _)                 = self.TARGET.points.shape
        
        self.standardize = standardize
        if self.standardize == True:
            #~ self.standard_scale, self.scaler_scan, self.scaler_target  = self.standardize_data()
            self.standard_scale, self.scan_mean, self.target_mean  = self.standardize_data()
        else:
            pass
        
        self.T_SCAN                 = copy.deepcopy(self.SCAN)
        
        if T_init is not None and T_init.shape[0] and T_init.shape[1] is not self.D+1:
            raise Exception("Initial guess must be a homogenous transformation matrix with data dimension plus 1!")
        else:
            pass
    
        self.T_init = numpy.eye(self.D+1) if T_init is None else T_init
        self.transformation = numpy.eye(self.D+1)
        self.time = 0.0
        # Apply initial guess
        if numpy.allclose(self.T_init, numpy.eye(self.T_init.shape[0])) is False: 
            self.T_SCAN.transform(transformation=self.T_init)
            self.transformation = copy.deepcopy(self.T_init)
        else:
            pass
    
        self.success                = False    


    def Horns_method(X, Y, weighting=False, weights=None):
        """
        Closed form method for aligning point set X onto Y when given correspondences. 
        This is used in the p2p ICP and also in RANSAC like algorithms.
        """
        D = X.shape[1]
        if Y.shape[1] != D:
            raise Exception("Point clouds are not of same dimension!")
            
        if weighting == True:
            weight_total = weights.sum()
            centroid_X = (1/weight_total*(weights.T @ X)).flatten()
            centroid_Y = (1/weight_total*(weights.T @ Y)).flatten()
            XX = X - centroid_X
            YY = Y - centroid_Y
            
            H = 1/weight_total*((weights[:]*XX[:]).T @ YY)
        else:
            centroid_X = numpy.mean(X, axis=0)
            centroid_Y = numpy.mean(Y, axis=0)
            XX = X - centroid_X
            YY = Y - centroid_Y
            H = XX.T @ YY 
        
        U, S, Vt = numpy.linalg.svd(H)
        R = Vt.T @ U.T            
        if numpy.linalg.det(R) < 0: # special reflection case
            Vt[D-1, :] *= -1
            R = Vt.T @ U.T
        t = centroid_Y.T - R @ centroid_X.T

        return R, t


    def find_nn(source, target, dist_metric="euclidean"):
        """ Find nearest neighbors in kD-tree search for each target point """
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto", metric=dist_metric, n_jobs=-1).fit(target)  
        nn_distances, nn_indices = nbrs.kneighbors(source)
        
        return nn_distances, nn_indices


    def random_draw(source, size_sample, return_idx=False):
        """ 
        Draw random elements from array 
        
        args:
            source (numpy array):   Source array
            size_sample (int):      Number of samples that are drawn
        """
        numpy.random.seed()
        ran_idx = numpy.random.choice(numpy.arange(source.shape[0]), replace=False, size=size_sample)
        ran_sample = source[ran_idx]
    
        if return_idx == False:
            return ran_sample
        else:
            return ran_sample, ran_idx


    def m_estimator(residuals, kernel="huber"):
        """ Huber Kernel """
        m_residuals = numpy.zeros(residuals.shape)
        threshold = 1.4826*numpy.median(residuals)
        
        if kernel == "huber":
            smaller_idx = numpy.where(residuals <= threshold)[0]
            m_residuals[smaller_idx] = residuals[smaller_idx]**2
            bigger_idx = numpy.where(residuals > threshold)[0]
            m_residuals[bigger_idx] = 2*threshold*abs(residuals[bigger_idx])-threshold**2
            
        elif kernel == "tukey":
            smaller_idx = numpy.where(residuals <= threshold)[0]
            m_residuals[smaller_idx] = (threshold**2)/6 * (1-(1-(residuals[smaller_idx]**2/threshold**2))**3)
            bigger_idx = numpy.where(residuals > threshold)[0]
            m_residuals[bigger_idx] = (threshold**2)/6
        
        else:
            raise Exception("Select (huber) or (tukey) as options for m-estimator function!")
        
        return m_residuals
    
    
    def standardize_data(self):
        """ Standardize data to unit variance and zero mean """    
        # Selfmade centering around mean
        target_mean = self.TARGET.center_around_origin(return_mean=True)
        scan_mean = self.SCAN.center_around_origin(return_mean=True)
        # In the original implementation we have only a scalar and scale each dimension the same
        target_scale = numpy.sqrt( (self.TARGET.points**2).sum()/self.N )
        scan_scale = numpy.sqrt( (self.SCAN.points**2).sum()/self.M )
        # Here we have a scalar scale
        if target_scale > scan_scale:
            scale = target_scale
        else:
            scale = scan_scale
            
        self.TARGET.points = self.TARGET.points/scale
        self.SCAN.points = self.SCAN.points/scale
        
        return scale, scan_mean, target_mean
    
    
    def destandardize_data(self):
        """ Destandardize the data with given sklearn.Standardscaler object """
        # Decenter both data sets
        self.TARGET.decenter_pcd(self.target_mean)
        self.SCAN.decenter_pcd(self.scan_mean)
    
        # We use the same scalar value to get back old variance 
        self.TARGET.points = self.TARGET.points * self.standard_scale
        self.SCAN.points = self.SCAN.points * self.standard_scale
    
        return

    
