import numpy
import copy
import ipdb
import dask.array
import h5py
from pyreg.local_reg.local_registration_base import LocalRigidRegistration
from pyreg.point_cloud_class import *

def contains_nan(array):
    """ With very low values, we can get errors for self.P because we are very close to zero """
    if numpy.isnan(array).any() == True:
        return True
    else:
        return False

class CpdRigidRegistration(LocalRigidRegistration):
    """
    Rigid registration of two points sets X(target) and Y(scan)
  
    Attributes:
        X: model point set
        Y: measured point set
        TY: Transformed point set
        N, M: Number of points
        D: Dimension of the Data
        R: Rotation matrix (Computed via SVD)
        t: translation
        scale: scaling parameter
        q: cost function
        iteration: Iteration count of the CPD
        sigma2: squared variance of GMM
        max_iterations: Maximum number of iterations
        tolerance: Stopping tolerance
        w: outliers or noise level estimate
    """
    # TODO Change dask computations: Dont write out P_dask, instead pass it to maximization and then compute all other variable on spot
    # This avoids communication between disk and RAM and just schedules the computations -> Speed up
    def __init__(self, SCAN, TARGET, output_path, T_init=None, ConvergenceCriteria=None, w=None, standardize=False, scaling=False):    
        """ Initializes the object(set_X, set_Y) """
        LocalRigidRegistration.__init__(self, SCAN, TARGET, T_init, ConvergenceCriteria, standardize)
        
        self.output_path    = output_path
        # We use X and Y for this algorithm because its easier to read
        self.X              = self.TARGET
        self.Y              = self.SCAN
        self.TY             = copy.deepcopy(self.Y)
        (self.N, self.D)    = self.X.points.shape
        (self.M, _)         = self.Y.points.shape
        
        self.scaling        = scaling
        self.scale          = 1.0 
        
        self.R              = numpy.eye(self.D) if T_init is None else T_init[:self.D, :self.D]
        self.t              = numpy.atleast_2d(numpy.zeros((1, self.D))) if T_init is None else T_init[:, -1]
        
        self.scale          = 1.0
        self.w              = 0.0 if w is None else w
        self.P              = numpy.zeros((self.M, self.N))
        
        if (self.M <= 30000 and self.N <= 30000) is True:
            #~ self.sigma2 = distance.cdist(self.Y.points, self.X.points, metric="sqeuclidean")
            #~ self.sigma2 = self.sigma2.sum()/(self.D*self.N*self.M)
            self.sigma2     = ((self.X.points[:, None, :] - self.Y.points[None, :, :]) ** 2).sum(axis=2).sum()
            self.sigma2     = self.sigma2/(self.D*self.M*self.N)
        else:
            target_dask     = dask.array.from_array(self.X.points, chunks=(1000, 3))
            scan_dask       = dask.array.from_array(self.Y.points, chunks=(1000, 3))
            sigma2_dask     = compute_sqeuclidean_dask(scan_dask, target_dask)
            sigma2_dask     = sigma2_dask.sum()/(self.D*self.N*self.M)
            self.sigma2     = sigma2_dask.compute()
        
        self.prev_err       = self.sigma2 
        self.err_history.append(self.sigma2)    
    
    
    def compute_sqeuclidean_dask(a, b):
        """
        Compute the squared distance between every point of 2 dask point arrays
        
        params:
            a, b        = N x D, m x D dask arrays    
        returns:
            dist_arr    = N x M dask array of all distance permuations
        """
        a_dim = a.ndim
        b_dim = b.ndim
        if a_dim == 1:
            a = a.dask.array(1, 1, a.shape[0])
        if a_dim >= 2:
            a = a.reshape(numpy.prod(a.shape[:-1]), 1, a.shape[-1])
        if b_dim > 2:
            b = b.reshape(numpy.prod(b.shape[:-1]), b.shape[-1])
        
        diff = a-b
        
        dist_arr = dask.array.einsum("ijk,ijk->ij", diff, diff)
        dist_arr = dask.array.squeeze(dist_arr)
        
        return dist_arr

    
    def register(self): 
        """ Main function of every registration algorithm """
        start = time.time()
        while self.iteration < self.max_iteration and self.err_delta > self.tolerance:
            self.iterate()
            self.iteration += 1    
            if abs(self.sigma2) < 1e-12:
                break
        
        if self.standardize == True:
            LocalRigidRegistration.destandardize_data(self)
            t = self.transformation[:-1, -1] 
            R = self.transformation[:self.D, :self.D] 
            self.transformation[:-1, -1] = self.standard_scale*t + self.target_mean.T - R @ self.scan_mean.T
        else:
            pass
        
        # TODO Change this with plausibilty criterion
        # This is depending on the data set
        #~ if self.inlier_rmse <= 1e-3:
            #~ self.success = True
        
        end = time.time()
        self.time = end-start
        
        return

    
    def iterate(self):
        """ Perform the CPD as an Expectation Maximization algorithm """
        start_iter = time.time()
        self._e_step()
        end_estimate = time.time()
        time_estimate = end_estimate-start_iter
        self.times["estimation"].append(time_estimate)
        
        # When variance is small we might get a zero division for some elements, the probability is then 0 as well
        if contains_nan(self.P) == True:
            mask = numpy.isnan(self.P)
            self.P[mask] = 0.0
        else:
            pass
    
        start_max = time.time()
        self._m_step()
        end_max = time.time()
        time_max = end_max - start_max
        self.times["maximization"].append(time_max)
        
        self.TY.points = self.scale * self.Y.points @ self.R.T + numpy.tile(self.t.T, (self.M, 1))
        
        self.err_delta = abs(self.sigma2 - self.prev_err)
        self.prev_err = self.sigma2
        self.err_history.append(self.sigma2)


    def _e_step(self):
        """ Expecation step """    
        if (self.M <= 20000 and self.N <= 20000) is True:
            numerator = self.X.points[:, numpy.newaxis, :] - self.TY.points
            numerator = numerator*numerator
            numerator = numpy.sum(numerator, 2)
            numerator = numpy.exp(-1.0/(2*self.sigma2)*numerator)
            num_sum = numpy.sum(numerator, 1)
            denominator = (num_sum + (2*numpy.pi*self.sigma2)**(self.D/2)*self.w/(1-self.w)*(float(self.M)/self.N)).reshape([self.N, 1])
            self.P = (numerator/denominator).T
        
        else:
            target_dask = dask.array.from_array(self.X.points, chunks=(1000, 3))
            scan_dask = dask.array.from_array(self.TY.points, chunks=(1000, 3))
            numerator_dask = compute_sqeuclidean_dask(scan_dask, target_dask)
            numerator_dask = numpy.exp(-1.0/(2*self.sigma2)*numerator_dask)
            num_sum = numpy.sum(numerator_dask, 1)
            denominator = (num_sum + (2*numpy.pi*self.sigma2)**(self.D/2)*self.w/(1-self.w)*(float(self.M)/self.N)).reshape([self.N, 1])
            P_dask = (numerator_dask/denominator)
            
            f = h5py.File(self.output_path+"/correspondences.hdf5")
            data = f.require_dataset("/data", shape=P_dask.shape, dtype=P_dask.dtype)
            dask.array.store(P_dask, data)
        
        # This is more accurate but slower!
        #~ P = np.zeros((self.M, self.N))
        #~ for i in range(0, self.M):
            #~ diff     = self.X - np.tile(self.TY[i, :], (self.N, 1))
            #~ diff     = np.multiply(diff, diff)
            #~ P[i, :]  = P[i, :] + np.sum(diff, axis=1)
        #~ c = (2 * np.pi * self.sigma2) ** (self.D / 2)
        #~ c = c * self.w / (1 - self.w)
        #~ c = c * self.M / self.N
        #~ P = np.exp(-P / (2 * self.sigma2))
        #~ den = np.sum(P, axis=0)
        #~ den = np.tile(den, (self.M, 1))
        #~ den[den==0] = np.finfo(float).eps
        #~ den += c
        #~ self.P   = np.divide(P, den)
        
        return
    
    
    def _m_step(self):
        """ Maximization step """
        if (self.M <= 20000 and self.N <= 20000) is True:
            p1 = (numpy.sum(self.P, 1)).reshape([self.M, 1])
            px = self.P @ self.X.points
            pt1 = (numpy.sum(self.P.T, 1)).reshape([self.N, 1])
        else:
            f = h5py.File(self.output_path+"correspondences.hdf5")
            dset = f["/data"]
            P_dask = dask.array.from_array(dset, chunks=(1000, 1000))
            
            p1 = (numpy.sum(P_dask, 1)).reshape([self.M, 1]).compute()
            px = (P_dask @ self.X.points).compute()
            pt1 = (numpy.sum(P_dask.T, 1)).reshape([self.N, 1]).compute()
        
        Np = numpy.sum(pt1)
        
        mu_x = self.X.points.T @ pt1 / Np
        mu_y = self.Y.points.T @ p1 / Np
        A = px.T @ self.Y.points - Np*(mu_x @ mu_y.T)
        
        [U, S, V] = numpy.linalg.svd(A)
        S = numpy.diag(S)
        C = numpy.eye(self.D)
        C[-1, -1] = numpy.linalg.det(U @ V)
        self.R = U @ C @ V
        if self.scaling == True:
            self.scale = numpy.trace(S @ C)/(sum(sum(self.Y.points*self.Y.points*numpy.matlib.repmat(p1, 1, self.D))) - Np *
                                            mu_y.T @ mu_y)
        else:
            self.scale = 1.0
        
        sigma22 = numpy.abs(sum(sum(self.X.points*self.X.points*numpy.matlib.repmat(pt1, 1, self.D)))-Np *
                        (mu_x.T @ mu_x)- self.scale*numpy.trace(S @ C))/(Np*self.D)
        self.sigma2 = sigma22[0][0]
        self.t = mu_x - self.scale*self.R @ mu_y
        self.transformation[:self.D, :self.D] = self.R
        self.transformation[:-1, -1] = self.t.flatten()
                
        # This might have the same accuracy but is slower
        #~ N_p = P.sum()
        #~ mu_x = 1 / N_p * X.T @ P.sum(axis=1)
        #~ mu_y = 1 / N_p * Y.T @ P.sum(axis=0)
        #~ XX = self.X.points - numpy.tile(mu_x.T, (self.N, 1))
        #~ YY = self.Y.points - numpy.tile(mu_y.T, (self.M, 1))
        #~ A = XX.T @ self.P.T @ YY
        #~ U, _, V_t = numpy.linalg.svd(A, full_matrices=True)
        #~ C = numpy.eye(self.D)
        #~ C[-1, -1] = numpy.linalg.det(U @ V_t)
        #~ self.R = U @ C @ V_t
        #~ self.scale = numpy.trace(A.T @ self.R) / numpy.trace((Y_hat.T * self.P.sum(axis=0, keepdims=True)) @ Y_hat)
        #~ self.t = muX - self.s * self.R @ muY
        
        return
        
