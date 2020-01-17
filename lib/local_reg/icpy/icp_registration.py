import numpy
import math
import ipdb
import time
import copy
from sklearn.neighbors import NearestNeighbors
from pyreg.local_reg.local_registration_base import LocalRigidRegistration
from pyreg.point_cloud_class import *
import pyreg.functions
import psutil
from scipy.spatial import distance
import sklearn.metrics  


def mean_squared_error(X, Y):
    """ 
    Compute the euclidean distance between all corresponding point pairs and the rmse of it
    """
    sq_dist = ((X - Y)**2).sum(axis=1)
    
    return numpy.sqrt(numpy.mean(sq_dist))
    

def get_process_memory():
    process = psutil.Process(os.getpid())
    
    return process.memory_info().rss


class IcpRegistration(LocalRigidRegistration):    
    """  
    Rigid registration of two points sets SCAN(set_p) and TARGET(set_q). Be careful, always use copies of the original clouds!
    Perform the rigid ICP on the object via register(). We register SCAN onto the target! 
    
    # TODO we are selecting from target, because the model is the moving point cloud in outlier test!
    # CHANGE tscan_idx to target_idx in rejection!
    
    Attributes:
        SCAN (ndarray)          : moving point set
        TARGET (ndarray)        : fixed point set 
        T_SCAN (ndarray)        : Transformed point set
        N (int)                 : Number of points in SCAN
        M (int)                 : Number of points in TARGET
        D (int)                 : Dimension of the Data (=3)
        R (ndarray)             : Rotation matrix
        t (ndarray)             : translation vector
        transformation (ndarray): total transformation
        weights (str)           : Weights for corresponding points
        inlier_rmse (float)     : Root Mean Square error
        tolerance (float)       : Stopping tolerance
        err_history (ndarray)   : Save the rmse over iterations for plot
    """
  
    def __init__(self, SCAN, TARGET, T_init, ConvergenceCriteria=None, Error_metric="point_to_point", 
                 Selection=None, Weighting=None, Rejection=None, LM=None, M_estimator=None):
        LocalRigidRegistration.__init__(self, SCAN, TARGET, T_init, ConvergenceCriteria)
        
        self.T = numpy.eye(self.D+1)
        self.error_metric = Error_metric
        if self.error_metric == "point_to_plane" or self.error_metric == "levenberg_marquardt":
            if self.TARGET.has_normals() == False:
                raise Exception("Input target has no normals! Precompute these or switch to point_to_point metric...")
            else:
                pass        
        
        self.rejection = Rejection
        if self.rejection.heuristics == None:
            self.rejection = None
        else:
            pass
        self.selection = Selection
        if self.selection.kind == None:
            self.selection = None
        else:
            pass
        self.weighting = Weighting 
        if self.weighting.kind != None:
            self.weights = numpy.ones((self.N, 1))
            if self.error_metric == "point_to_plane" or self.error_metric == "levenberg_marquardt":
                raise Warning("Weighting is currently only possible with point-to-point metric!")
            else:
                pass
        else:
            self.weighting = None
        
        self.correspondence_set     = numpy.zeros((self.N, 2))
        self.prev_err               = 0.0
        self.times["nn_search"]     = []
        
        # We have to do one step in advance to have the current parameters upon ICP termination
        time_estimate = self._estimation_step()
        self.times["estimation"].append(time_estimate)
        
        if self.weighting != None:
            weight_total = self.weights.sum()
            self.inlier_rmse = numpy.sqrt((1/weight_total*self.weights).T @ self.inlier_deviations**2)
        else:
            self.inlier_rmse = numpy.sqrt(numpy.mean(self.inlier_deviations**2))
        self.err_history.append(self.inlier_rmse)
        
        # We want to monitor the true rmse when doing controlled experiments with same set sizes
        #~ self.true_err_history = []
        #~ self.true_rmse = mean_squared_error(self.TARGET.points, self.T_SCAN.points)
        #~ self.true_err_history.append(self.true_rmse)
        
        if self.error_metric == "levenberg_marquardt":
            self.mu = LM.lambda_init
            self.multiplier = LM.multiplier

    
    def register(self): 
        """ Main function of every local registration algorithm """
        start = time.time()
        while self.iteration < self.max_iteration and self.err_delta > self.tolerance:
            self.iterate()
            self.iteration += 1
        
        if self.standardize == True:            
            LocalRigidRegistration.destandardize_data(self)
            t = self.transformation[:-1, -1] 
            R = self.transformation[:self.D, :self.D] 
            self.transformation[:-1, -1] = self.standard_scale*t + self.target_mean.T - R @ self.scan_mean.T
        else:
            pass
        
        # TODO this is data specific
        if self.inlier_rmse <= 1e-4:
            self.success = True
        
        end = time.time()
        self.time = end-start
        
        return
    
    
    def iterate(self):
        """ Perform the ICP as an Expectation Maximization algorithm """
        start_iter = time.time()
        
        time_max = self._maximization_step()
        self.times["maximization"].append(time_max)
        
        #~ self.T_SCAN.transform(transformation=self.T)
        self.T_SCAN = copy.deepcopy(self.SCAN)
        self.T_SCAN.transform(transformation=self.transformation)
        
        self.correspondence_set = numpy.zeros((self.N, 2)) # Reset the correspondence set
        time_estimate = self._estimation_step()
        self.times["estimation"].append(time_estimate)
        
        if self.weighting != None:
            weight_total = self.weights.sum()
            self.inlier_rmse = numpy.sqrt((1/weight_total*self.weights).T @ self.inlier_deviations**2)
        else:
            self.inlier_rmse = numpy.sqrt(numpy.mean(self.inlier_deviations**2)) 
        
        self.err_delta = abs(self.inlier_rmse - self.prev_err)
        self.prev_err = self.inlier_rmse
        self.err_history.append(self.inlier_rmse)
        
        # For controlled experiments with same set sizes
        #~ self.true_rmse = mean_squared_error(self.TARGET.points, self.T_SCAN.points)
        #~ self.true_err_history.append(self.true_rmse)
        
        end_iter = time.time()
        time_iter = end_iter - start_iter
        self.times["iteration"].append(time_iter)
        
        return


    def _estimation_step(self):
        """ Estimate correspondences """
        def _select_points(selection_object):
            """ Select/Sample points out of SCAN """
            if selection_object.kind == "random":
                num_sel = int(math.floor(selection_object.samples*self.N))
                T_SCAN_selected = copy.deepcopy(self.T_SCAN)
                points_sample, rand_idx = LocalRigidRegistration.random_draw(source=T_SCAN_selected.points, 
                                                                                      size_sample=num_sel, 
                                                                                      return_idx=True
                                                                                    )
                T_SCAN_selected.points = points_sample
                if self.T_SCAN.has_normals() == True:
                    T_SCAN_selected.normals = T_SCAN_selected.normals[rand_idx, :]
                else:
                    pass
                
                return T_SCAN_selected, rand_idx
                
            else:
                raise ValueError("Selection has to be str(random)")
    
    
        def _reject_matches(nn_distances, tscan_idx, nn_indices, rejection_object):
            """
            Reject some of the matches with a quality criteria 
    
            rejection should be either ["fixed", percentage] or ["distance", max_distance] 
            """
            if rejection_object.kind == "trimmed":
                """ Use only the best x % of matches """
                length = int(math.floor((1-rejection_object.trim_percentage)*T_SCAN.points.shape[0]))
                
                distances_sorted_idx = nn_distances.argsort(axis=0) 
                best_match_idx = distances_sorted_idx[:length].flatten()
                nn_distances_match = nn_distances[best_match_idx] # non rejected distances
                tscan_match_idx = tscan_idx[best_match_idx].flatten()
                target_match_idx = nn_indices[best_match_idx].flatten() 
                
                return nn_distances_match, tscan_match_idx, target_match_idx
        
            if rejection_object.kind == "iqr":
                """ Detect outliers by applying trimmed statistics """
                q75, q25 = numpy.percentile(nn_distances, [75 ,25])
                iqr = q75 - q25 
                max_dist = q75 + rejection_object.iqr_factor*iqr
      
                best_match_idx =  numpy.where(nn_distances < max_dist)[0]
                target_match_idx = nn_indices[best_match_idx].flatten() 
                tscan_match_idx = tscan_idx[best_match_idx].flatten()
                nn_distances_match = nn_distances[best_match_idx] # non rejected distances
                
                return nn_distances_match, tscan_match_idx, target_match_idx
      
            elif rejection_object.kind == "max_distance":
                """ Maximum distance criteria """
                best_match_idx = numpy.where(nn_distances < rejection_object.max_distance)[0]
                target_match_idx = nn_indices[best_match_idx].flatten() 
                tscan_match_idx = tscan_idx[best_match_idx].flatten()
                
                nn_distances_match = nn_distances[best_match_idx] # non rejected distances
                # This is not very smart, because the Code exits, we dont need the attributes anymore
                if nn_distances_match.size == 0:
                    self.inlier_rmse =  numpy.mean(nn_distances) 
                    self.err_delta = abs(self.inlier_rmse - self.prev_err)
                    self.prev_err = self.inlier_rmse
                    
                    raise Exception("All matches were rejected! Change MaximumDistance criterion in config file!")            
                
                else:
                    pass

                return nn_distances_match, tscan_match_idx, target_match_idx
    
            else:
                raise ValueError("Rejection should be either str(fixed), str(distance) or str(trimmed)")


        def _weight_matches(nn_distances, weighting_object):
            """ Weight the matches acording to weighting criterion """    
            self.weights = numpy.ones((nn_distances.shape[0], 1)) 
            
            if weighting_object.kind == "huber":
                threshold = 1.4826*numpy.median(nn_distances) # This is the MAD value
                smaller_idx = numpy.where(nn_distances <= threshold)[0]
                self.weights[smaller_idx] = 1
                bigger_idx = numpy.where(nn_distances > threshold)[0]
                self.weights[bigger_idx] = threshold/abs(nn_distances[bigger_idx])    
            
            elif weighting_object.kind == "tukey":
                threshold = 1.4826*numpy.median(nn_distances) # This is the MAD value
                smaller_idx = numpy.where(nn_distances <= threshold)[0]
                self.weights[smaller_idx] = (1-(nn_distances*2/threshold**2))**2
                bigger_idx = numpy.where(nn_distances > threshold)[0]
                self.weights[bigger_idx] = 0
            
            elif weighting_object.kind == "linear":
                """ linear distance weights """
                dist_max = numpy.amax(nn_distances, axis=0)
                self.weights = 1 - nn_distances/dist_max
            
            elif weighting_object.kind == "gaussian":
                sigma2 = numpy.std(nn_distances)**2
                self.weights = 1/numpy.sqrt(2*numpy.pi*sigma2)*numpy.exp(-(nn_distances**2/(2*sigma2)))
            
            elif weighting_object.kind == "normals":
                """ normal compatibility """
                if self.SCAN.has_normals() == False or self.TARGET.has_normals == False:
                    raise Exception("We have no normals to compute compatibility!") 
                else:
                    t_scan_normals = self.T_SCAN.normals[self.correspondence_set[:, 0].flatten()]
                    target_normals = self.TARGET.normals[self.correspondence_set[:, 1].flatten()]
                    t_scan_normals = t_scan_normals.reshape((t_scan_normals.shape[0], t_scan_normals.shape[1], 1))
                    target_normals = target_normals.reshape((target_normals.shape[0], target_normals.shape[1], 1))
                    
                    self.weights = (numpy.transpose(t_scan_normals, axes=(0, 2, 1)) @ target_normals).flatten()
                    self.weights = self.weights.reshape(self.weights.shape[0], 1)
                    # Reflections! If normals are flipped and not correct during the PCA, we have negative weights!
                    self.weights = numpy.negative(self.weights, where=self.weights<0, out=self.weights)

            else:
                raise ValueError("Weighting should be either str(normalized), str(normals)")
        

        start = time.time() # Estimation time
    
        if self.selection != None:
            TARGET = copy.deepcopy(self.TARGET)
            T_SCAN, tscan_idx = _select_points(self.selection)
        else:
            TARGET = copy.deepcopy(self.TARGET)
            T_SCAN = copy.deepcopy(self.T_SCAN)
            tscan_idx = numpy.arange(T_SCAN.points.shape[0])
        
        nn_start = time.time()
        nn_distances, nn_indices = LocalRigidRegistration.find_nn(T_SCAN.points, TARGET.points)
        nn_end = time.time()
        nn_time = nn_end-nn_start
        self.times["nn_search"].append(nn_time)
        
        if self.rejection != None:
            nn_distances_match, tscan_match_idx, target_match_idx = nn_distances, tscan_idx, nn_indices
            for kind in self.rejection.heuristics:
                self.rejection.kind = kind
                nn_distances_match, tscan_match_idx, target_match_idx = _reject_matches(nn_distances_match, tscan_match_idx, 
                                                                                        target_match_idx, self.rejection)
        else:
            tscan_match_idx = tscan_idx
            target_match_idx = nn_indices
            nn_distances_match = nn_distances

        self.inlier_deviations = nn_distances_match
        self.correspondence_set = self.correspondence_set[0:nn_distances_match.shape[0]]
        self.correspondence_set[:, 0] = tscan_match_idx
        self.correspondence_set[:, 1] = target_match_idx.flatten()
        self.correspondence_set = numpy.int_(self.correspondence_set)
        self.overlap = nn_distances_match.shape[0]/self.N
      
        if self.weighting != None:
            _weight_matches(nn_distances_match, self.weighting)
        else:
            pass
        
        end = time.time()
        time_estimate = end-start
        
        return time_estimate


    def _maximization_step(self):
        """ 
        Compute the transformation based on the found corresponding points and computed weights
        """
        start = time.time()
        
        if self.error_metric == "point_to_point":
            """
            Point to Point error metric optimization with SVD approach
            
            We align SCAN onto TARGET (X onto Y)
            """            
            T_SCAN_matched = self.T_SCAN.points[self.correspondence_set[:, 0].flatten()] 
            TARGET_matched = self.TARGET.points[self.correspondence_set[:, 1].flatten()]
            
            if self.weighting == None:
                R, t = LocalRigidRegistration.Horns_method(T_SCAN_matched, TARGET_matched)
            else:
                R, t = LocalRigidRegistration.Horns_method(T_SCAN_matched, TARGET_matched, weighting=True, weights=self.weights)
            self.T[:self.D, :self.D] = R
            self.T[:self.D, self.D] = t
            self.transformation = self.T @ self.transformation # total transform
            
            end = time.time()
            time_max = end-start
            
            return time_max
        
        
        elif self.error_metric == "point_to_plane":
            """ This approach uses a linearized transformation matrix to solve the nonlinear minimization problem """
            # Linearized Rotation matrix
            T_SCAN_matched = PointCloud(points=self.T_SCAN.points[self.correspondence_set[:, 0].flatten()])
            TARGET_matched = PointCloud(points=self.TARGET.points[self.correspondence_set[:, 1].flatten()], 
                                        normals=self.TARGET.normals[self.correspondence_set[:, 1].flatten()])
            if TARGET_matched.points.shape[0] < 3:
                raise Exception("Not enough matches found for computation! Change rejection criterion!")
            else:
                pass 
            
            Jr_sum = numpy.zeros((6, 1))
            JJT_sum = numpy.zeros((6, 6))
            dim1, dim2 = TARGET_matched.points.shape[0], TARGET_matched.points.shape[1] # N and 3
            J = numpy.zeros((dim1, 6, 1))
            Jr = numpy.zeros((dim1, 6, 1))
            
            TARGET_matched.points = TARGET_matched.points.reshape((dim1, dim2, 1))
            T_SCAN_matched.points = T_SCAN_matched.points.reshape((dim1, dim2, 1))
            TARGET_matched.normals = TARGET_matched.normals.reshape((dim1, dim2, 1))
            tangents = numpy.cross(T_SCAN_matched.points[:, :, 0], TARGET_matched.normals[:, :, 0])
            diff = T_SCAN_matched.points - TARGET_matched.points 
            residuals = diff.reshape((dim1, 1, dim2)) @ TARGET_matched.normals # this is r_i
            
            J[:] = numpy.hstack((tangents[:], TARGET_matched.normals[:].reshape((dim1, dim2)))).reshape((dim1, -1, 1))
            # This is the right hand side: 
            Jr = J[:]*residuals[:]
            # This is the left hand side:
            JJT = J @ J.reshape(dim1, 1, 6)    
            Jr_sum = Jr.sum(axis=0)
            JJT_sum = JJT.sum(axis=0)
            
            psd_test = pyreg.functions.is_matrix_psd(JJT_sum)
            if psd_test == True:
                # Use Cholesky decomposition to solve linear system: JJT_sum*(x) = -Jr_sum
                L = numpy.linalg.cholesky(JJT_sum) # 1. Cholesky Decomposition 
                y = numpy.linalg.solve(L, -Jr_sum) # 2. Solve for y with L*(y) = b
                result_vector = numpy.linalg.solve(L.T.conj(), y).flatten() # Solve for x with L.H*(x) = y
            else:
                # Use more robust, but slower LU Decomposition when JJT_sum is not positive definite
                P, L, U = scipy.linalg.lu(JJT_sum)
                y = numpy.linalg.solve(L, P @ -Jr_sum)
                result_vector = numpy.linalg.solve(U, y).flatten()
    
            # Get transformation matrix from 6D solution vector
            R = pyreg.functions.get_matrix_from_xyz_euler_angles(result_vector[:self.D])
            self.T[:self.D, :self.D] = R
            self.T[:self.D, -1] = result_vector[self.D:]            
            self.transformation = self.T @ self.transformation # total transform
            
            end = time.time()
            time_max = end-start
    
            return time_max

    
        elif self.error_metric == "levenberg_marquardt":
            def compute_residuals(X, Y):
                """ point-to-plane reisudals """
                M = X.points.shape[0]
                D = X.points.shape[1]
                Y.points = Y.points.reshape((M, D, 1))
                X.points = X.points.reshape((M, D, 1))
                Y.normals = Y.normals.reshape((M, D, 1))
                diff = X.points - Y.points 
                residuals = diff.reshape((M, 1, D)) @ Y.normals 

                return residuals
            
            
            def compute_jacobian(SCAN_matched, TARGET_matched, current_params, residuals):
                """ Compute the Jacobian with the current parameters """
                rot = current_params[:self.D]
                grad_rot = pyreg.functions.get_current_xyz_rotation_gradient(rot)
                M = T_SCAN_matched.points.shape[0]
                
                grad_rot = numpy.broadcast_to(grad_rot, (M, 9, 3)).reshape((M, 3, 3, 3))
                J1 = grad_rot[:, 0, :, :] @ SCAN_matched.points.reshape((M, 3, 1))
                J1 = numpy.transpose(J1, (0, 2, 1)) @ TARGET_matched.normals.reshape((M, 3, 1))
                J2 = grad_rot[:, 1, :, :] @ SCAN_matched.points.reshape((M, 3, 1))
                J2 = numpy.transpose(J2, (0, 2, 1)) @ TARGET_matched.normals.reshape((M, 3, 1))
                J3 = grad_rot[:, 2, :, :] @ SCAN_matched.points.reshape((M, 3, 1))
                J3 = numpy.transpose(J3, (0, 2, 1)) @ TARGET_matched.normals.reshape((M, 3, 1))
                
                upper_J = numpy.hstack((J1, J2, J3))
                lower_J = TARGET_matched.normals
                J = numpy.hstack((upper_J, lower_J))
                
                return J
            
            
            def compute_m_estimator(residuals, kernel="huber"):
                if kernel == "huber":
                    k = 1.4826*numpy.median(abs(residuals)) # This is the Median of Absolute Deviation (MAD)
                    smaller_idx = numpy.where(abs(residuals) <= k)[0]
                    bigger_idx = numpy.where(abs(residuals) > k)[0]
                    weights = copy.deepcopy(residuals)
                    
                    weights[smaller_idx] = 1
                    test = numpy.where(abs(residuals[bigger_idx]) < 1.0e-12)[0]
                    if test.shape[0] == residuals[bigger_idx].shape[0]: # All elements are below
                        weights[bigger_idx] = 1
                    else:
                        weights[bigger_idx] = k/abs(residuals[bigger_idx])
                    
                    return weights
                
                else:
                    raise Exception("No other kernel supported currently!")
            
           
            def compute_update_parameters(J, lambda_const, residuals):
                JTJ = J.T @ J
                update_params = -(numpy.linalg.pinv((JTJ)+lambda_const*numpy.eye(6))) @ J.T @ residuals.reshape(M, 1)
                # modified for scale invariance
                #~ update_params = -(numpy.linalg.pinv((JTJ)+lambda_const*numpy.diagonal(JTJ))) @ J.T @ residuals.reshape(M, 1)
                update_params = update_params.flatten()
            
                return update_params

            
            def evaluate_parameters(X, Y, params):  
                new_transform = numpy.eye(self.D+1)
                new_transform[:self.D, :self.D] = pyreg.functions.get_matrix_from_xyz_euler_angles(params[:self.D])
                new_transform[:self.D, -1] = params[self.D:]
                X.transform(transformation=new_transform)
                new_residuals = compute_residuals(X, Y)
                new_error = (new_residuals**2).sum()
                
                return new_error
            
        
            def adapt_damping(old_error, new_error, new_params, update_params, J, residuals, 
                              SCAN_matched, TARGET_matched, method="levenberg"):
                """ Adapt damping parameter """
                if method == "levenberg":
                    # Check if Gauss-Newton is better
                    smaller_mu = self.mu/self.multiplier
                    faster_update = compute_update_parameters(J, smaller_mu, residuals)
                    faster_params = current_params + faster_update
                    faster_error = evaluate_parameters(copy.deepcopy(SCAN_matched), copy.deepcopy(TARGET_matched), faster_params)
                    
                    if faster_error <= old_error:
                        best_params = faster_params
                        self.mu = smaller_mu
                    elif faster_error > old_error and new_error <= old_error:
                        best_params = new_params
                    else:
                        # Switch to Gradients Descent -> increase self.mu until the error gets reduced
                        i_temp = 0
                        bigger_mu = copy.deepcopy(self.mu)
                        slower_error = copy.deepcopy(new_error)
                        while slower_error > old_error and i_temp < 10:
                            bigger_mu = bigger_mu*self.multiplier
                            slower_update = compute_update_parameters(J, bigger_mu, residuals)
                            slower_params = current_params + slower_update
                            slower_error = evaluate_parameters(copy.deepcopy(SCAN_matched), copy.deepcopy(TARGET_matched), slower_params)
                            i_temp +=1
                
                        best_params = slower_params
                        self.mu = bigger_mu
                
                elif method == "gain":            
                # This is from a numerics book and uses the gain to adapt damping 
                    g = J.T @ residuals.reshape(M, 1)
                    nominator = old_error - new_error
                    update_params = update_params.reshape((6, 1))
                    denominator = update_params.T @ (self.mu*update_params - g)
                    gain = nominator / denominator
                    if gain > 0:
                        best_params = new_params
                        self.mu = self.mu*numpy.maximum(1/3, 1-(2*gain-1)**3)
                        self.multiplier = 2
                    else:
                        i_temp = 0
                        bigger_mu = copy.deepcopy(self.mu)
                        slower_error = copy.deepcopy(new_error)
                        slower_params = copy.deepcopy(new_params)
                        while slower_error > old_error and i_temp < 10:
                            self.mu = self.mu*self.multiplier
                            self.multiplier = self.multiplier*2
                            slower_update = compute_update_parameters(J, self.mu, residuals)
                            slower_params = current_params + slower_update
                            slower_error = evaluate_parameters(copy.deepcopy(SCAN_matched), copy.deepcopy(TARGET_matched), slower_params)
                            i_temp +=1
                            
                        best_params = slower_params
                
                else:
                    raise Exception("Choose a damping strategy, use (classic) or (gain)!")
                
                
                return best_params
            
            SCAN_matched = PointCloud(points=self.SCAN.points[self.correspondence_set[:, 0].flatten()])
            T_SCAN_matched = PointCloud(points=self.T_SCAN.points[self.correspondence_set[:, 0].flatten()])
            TARGET_matched = PointCloud(points=self.TARGET.points[self.correspondence_set[:, 1].flatten()], 
                                        normals=self.TARGET.normals[self.correspondence_set[:, 1].flatten()])
            if TARGET_matched.points.shape[0] < 3:
                raise Exception("Not enough matches found for computation! Change rejection criterion!")
            M = T_SCAN_matched.points.shape[0]
            
            rot = pyreg.functions.get_xyz_angles_from_matrix(self.transformation[:self.D, :self.D])
            trans = self.transformation[:self.D, -1]
            current_params = numpy.hstack((rot, trans))
            
            residuals = compute_residuals(T_SCAN_matched, TARGET_matched) # Use T_SCAN for computing residuals
            old_error = (residuals**2).sum()
            J = compute_jacobian(SCAN_matched, TARGET_matched, current_params, residuals) # Use SCAN for computing Jacobian
            J = J.reshape((M, 6))
            
            # Levenberg-Marquardt
            update_params = compute_update_parameters(J, self.mu, residuals)
            new_params = current_params + update_params
            # Use SCAN and new transform for computing new residuals
            new_error = evaluate_parameters(copy.deepcopy(SCAN_matched), TARGET_matched, new_params)
            
            best_params = adapt_damping(old_error, new_error, new_params, update_params, J, residuals, SCAN_matched, TARGET_matched)
            
            # Get transformation matrix from 6D solution vector
            best_rot = pyreg.functions.get_matrix_from_xyz_euler_angles(best_params[:self.D])
            best_trans = best_params[self.D:]
            self.transformation[:self.D, :self.D] = best_rot
            self.transformation[:self.D, -1] = best_trans
            
            end = time.time()
            time_max = end-start
    
            return time_max
        
        else:
            raise Exception("Something went wrong! No error metric was specified...")
    

class Rejection(object):
    """ 
    Rejection object contains parameters for icp options
    
    args:
        heuristic:      [list] list of heuristics that will be used
        kind:           [str] rejection heuristic
        trim_perc:      [float] % of points that are kept
        max_dist:       [float] maximum distance of point pairs
        iqr_fact:       [float] const multiplyer of the IQR
    """
    def __init__(self, kind="trimmed", trim_percentage=0.0, max_distance=None, iqr_factor=None, heuristics=None):
        self.heuristics = heuristics
        self.kind = kind
        self.trim_percentage = trim_percentage
        self.max_distance = max_distance
        self.iqr_factor = iqr_factor


class Selection(object):
    """
    Selection object contains parameters for the selection step
    
    args:
        kind:       [str] random 
        samples:    [float] percentage of sample points
    """
    def __init__(self, kind=None, samples=None):
        self.kind = kind
        self.samples = samples


class Weighting(object):
    """ 
    Weighting object contains parameters for weighting matches
    
    args:
        kind:       [str] weighting scheme
    """
    def __init__(self, kind=None):
        self.kind = kind


class LevenbergMarquardt(object):
    """ 
    Small class for LM-parameters 
    
    args:   
        lambda_init (float):         initial damping value
        multiplier (float):     const factor that gets lambda gets multiplied or divided by
    """
    def __init__(self, lambda_init=0.001, multiplier=2):
        self.lambda_init = lambda_init
        self.multiplier = multiplier


