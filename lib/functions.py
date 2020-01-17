import yaml
import os
import math
import numpy 
import pickle
import io
import matplotlib.pyplot
from scipy.sparse.linalg import arpack
from contextlib import contextmanager
import logging
from logging.handlers import TimedRotatingFileHandler
from pyreg.point_cloud_class import *


def contains_nan(array):
    """ With very low values, we can get errors for self.P because we are very close to zero """
    if numpy.isnan(array).any() == True:
        return True
    else:
        return False


def is_matrix_psd(A, tol = 1e-8):
                """ Is matrix A positive definite? If yes, then the eigenvalues are > 0 """
                vals, vecs = arpack.eigsh(A, k = 2, which = 'BE') # return the ends of spectrum of A
                
                return numpy.all(vals > -tol)


def get_mean_nn_distance(point_cloud):
    nbrs = NearestNeighbors(n_neighbors=2, algorithm="kd_tree", metric="euclidean", n_jobs=-1).fit(point_cloud.points)  
    nn_distances, nn_indices = nbrs.kneighbors(point_cloud.points)
    
    return numpy.mean(nn_distances[:, 1])


def get_nn_distances(pcd1, pcd2):
    """ Get distances from nn_neighbor of pcd1 to points in pcd2 """
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree", metric="euclidean", n_jobs=-1).fit(pcd1.points)  
    nn_distances, nn_indices = nbrs.kneighbors(pcd2.points)
    
    return nn_distances.flatten()


def get_rotation_difference(matrix1, matrix2, metric="kuffner"):
    """ 
    Get the difference in rotation with a quaternion metric
    
    args:
        metric:     ravani-roth     maps differences into range [0, sqrt(2)]
                    kuffner         maps differences into range [0, 1]
                    frobenius       maps differences into range [0,?], introduces singularities
    
        matrices:   3 dimensional rotation matrices
    
    returns: 
        scalar metric value for rotational difference
    """ 
    q1 = rotation_matr_to_quaternion(matrix1)
    q2 = rotation_matr_to_quaternion(matrix2)
    
    if metric == "ravani-roth":
        d1 = numpy.linalg.norm(q1 + q2)
        d2 = numpy.linalg.norm(q1 - q2)
        distance = numpy.minimum(d1, d2) 
    
    elif metric == "kuffner":
        distance = 1 - numpy.linalg.norm(q1 @ q2) 
        
    elif metric == "frobenius":
        diff = matrix1 - matrix2
        distance = numpy.linalg.norm(diff)

    else:
        raise Exception("Wrong error metric selected!")

    return distance
    

def rotation_matr_to_quaternion(matrix):
    """ 3x3 rotation matrix to quaternion """
    q = numpy.zeros(4)
    
    q[0] = 0.5*numpy.sqrt(1+numpy.trace(matrix))
    if numpy.allclose(0.0, q[0] , 1e-15, 1e-8) == False:
        q[1] = 1/(4*q[0])*(matrix[2, 1] - matrix[1, 2])
        q[2] = 1/(4*q[0])*(matrix[0, 2] - matrix[2, 0])
        q[3] = 1/(4*q[0])*(matrix[1, 0] - matrix[0, 1])
    
    else :
        q[1] = 0.5*numpy.sqrt(1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        
        if numpy.allclose(0.0, q[1] , 1e-15, 1e-8) == False:
            q[2] = 1/(4*q[1])*(matrix[0, 1] + matrix[1, 0])
            q[3] = 1/(4*q[1])*(matrix[0, 2] + matrix[2, 0])
            q[0] = 1/(4*q[1])*(matrix[2, 1] + matrix[1, 2])
        else:
            raise Exception("Denominator close to zero. Instability!")
        
    return q    


def generate_random_rotation_euler():
    """ 
    Angle has to be drawn from non-uniform distribution with propability density: 2/pi*sin²(0.5*phi)
    Then draw a random axis pointing to the surface of a unit 2-sphere
    """
    def pdf(x):
        """ Probability density function of euler angle for random rotations """
        return 2/numpy.pi*(numpy.sin(0.5*x))**2
        
    def cdf(x):
        """ Cumulative probability density function of euler-angle for random rotations """
        return (x-numpy.sin(x))/numpy.pi

    def generate_random_angle_with_cdf_distribution(cdf):
        """ 
        Use the inverse from the cdf to project random samples from a uniform distribution with interval [0, 1] 
        onto the wanted distribution
        """
        numpy.random.seed()
        n = 10000
        x = numpy.linspace(0, numpy.pi, n) 
        discrete_cdf = cdf(x)
        icdf = scipy.interpolate.interp1d(discrete_cdf, x, kind="cubic")
        x_rand = numpy.random.uniform() # Draw random value from [0, 1]
        rand_angle = icdf(x_rand) # Project uniform probability onto given distribution
        
        return rand_angle

    def generate_random_points_on_sphere():
        """ We generate a vector pointing to the surface of a unit sphere with random 3D coordinates """
        
        def normalize(v):
            norm = numpy.linalg.norm(v)
            if norm == 0: 
                return v
            else:
                return v/norm
        
        v = numpy.zeros(3)
        for i in range(3):
            k = 15 # We use the central limit theorem for our samples
            numpy.random.seed()
            k_sample = numpy.random.standard_normal(size=k)
            ran_coord = k_sample.sum()
            v[i] = ran_coord
        
        v = normalize(v)

        return v
    
    def get_rotation_matrix(axis, theta):
        """
        Derive R from axis and angle with exponential expansion:
        
        This is the same as below but slower and as a one liner
        """
        return scipy.linalg.expm(numpy.cross(numpy.eye(3), axis/scipy.linalg.norm(axis)*theta))
    
    ran_angle = generate_random_angle_with_cdf_distribution(cdf)
    ran_v = generate_random_points_on_sphere()
    
    K = numpy.asarray([[0, -ran_v[2], ran_v[1]],
                       [ran_v[2], 0, -ran_v[0]],
                       [-ran_v[1], ran_v[0], 0]])
    
    R = numpy.eye(3) + numpy.sin(ran_angle)*K + (1-numpy.cos(ran_angle))*(K @ K) 
    
    return R
    
  
def generate_random_rotation_quaternion():
    """ 
    Draw 4 values from normal distribution and normalize for unit quaternion
    """
    def gen_random_quaternion():
        """ Generate a random rotation using unit quaternions """
        def normalize(q):
            q_sum = numpy.linalg.norm(q)
            
            return q/q_sum 
        
        q = numpy.zeros(4)
        for i in range(4):
            k = 15 # We use the central limit theorem for our samples
            numpy.random.seed()
            k_sample = numpy.random.standard_normal(size=k)
            ran_coord = k_sample.sum()
            q[i] = ran_coord
        
        q = normalize(q)
        
        return q

    def quaternion_to_matrix(q):
        """ 
        Derive a homogenous rotation matrix from unit quaternion
        """
        R = numpy.asarray([[1.0-2.0*(q[2]**2+q[3]**2), 2.0*(q[1]*q[2]-q[3]*q[0]), 2.0*(q[1]*q[3]+q[2]*q[0])], 
                           [2.0*(q[1]*q[2]+q[3]*q[0]), 1.0-2.0*(q[1]**2+q[3]**2), 2.0*(q[2]*q[3]-q[1]*q[0])],
                           [2.0*(q[1]*q[3]-q[2]*q[0]), 2.0*(q[2]*q[3]+q[1]*q[0]), 1.0-2.0*(q[1]**2+q[2]**2)]])
        
        return R

    q = gen_random_quaternion()
    R = quaternion_to_matrix(q)
    
    return R


def get_memory():
    """
    Get node total memory and memory usage
    """
    with open("/proc/meminfo", "r") as mem:
        ret = {}
        tmp = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) == "MemTotal:":
                ret["total"] = int(sline[1])
            elif str(sline[0]) in ("MemFree:", "Buffers:", "Cached:"):
                tmp += int(sline[1])
        ret["free"] = tmp
        ret["used"] = int(ret["total"]) - int(ret["free"])
    
    return ret
    
    
def get_process_memory():
    process = psutil.Process(os.getpid())
    
    return process.memory_info().rss
  

def save_pickle(obj, filepath):
    """ Use pickle to save results in a binary format """
    with open(filepath+".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    return


def load_pickle(filepath):
    """ Use pickle to load results """
    with open(filepath+".pkl", "rb") as f:
        
        return pickle.load(f)
    

def create_timed_rotating_log(path):
    logger = logging.getLogger("run.log")
    logger.setLevel(logging.INFO)
 
    handler = TimedRotatingFileHandler(path, interval=1, backupCount=0)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


@contextmanager
def show_complete_array():
    oldoptions = numpy.get_printoptions()
    numpy.set_printoptions(threshold=numpy.inf)
    try:
        yield
    finally:
        numpy.set_printoptions(**oldoptions)


def read_config(config_name):
    """ Read in the yaml config file """
    with open(config_name, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
    
    return config


def write_config(config_data, path):
    """ Write the config file to specificied location """
    try:
        with io.open(path, "w", encoding="utf8") as outfile:
            yaml.dump(config_data, outfile, default_flow_style=False, allow_unicode=True)
    finally:
        pass
    
    return


def make_dir(path):
    """ If directory does not exist, create one """
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass
        
    return


def get_matrix_from_xyz_euler_angles(xyz_euler_angles):
    """ Convert XYZ Euler angles to active rotation matrix """
    c1, s1 = numpy.cos(xyz_euler_angles[0]), numpy.sin(xyz_euler_angles[0])
    c2, s2 = numpy.cos(xyz_euler_angles[1]), numpy.sin(xyz_euler_angles[1])
    c3, s3 = numpy.cos(xyz_euler_angles[2]), numpy.sin(xyz_euler_angles[2])
    R = numpy.asarray([[c2*c3, -c2*s3, s2],
                        [c1*s3+c3*s1*s2, c1*c3-s1*s2*s3, -c2*s1],
                        [s1*s3-c1*c3*s2, c3*s1+c1*s2*s3, c1*c2]])
    
    return R


def get_xyz_angles_from_matrix(R):
    """ Convert active rotation matrix into XYZ-euler angles """
    # ISSUE This does not account for gimbal lock!
    #~ x = math.atan2(-R[1, 2], R[2, 2])
    #~ z = math.atan2(-R[0, 1], R[0, 0])
    #~ cy = math.sqrt(R[0, 0]**2+R[0, 1]**2)
    #~ y = math.atan2(R[0, 2], cy)
    
    # New approach!
    x = math.atan2(-R[1, 2], R[2, 2])
    cy = math.sqrt(R[0, 0]**2+R[0, 1]**2)
    y = math.atan2(R[0, 2], cy)
    sx, cx = math.sin(x), math.cos(x)
    z = math.atan2(cx*R[1, 0]+sx*R[2, 0], cx*R[1, 1]+sx*R[2, 1])
    
    return numpy.array([x, y, z])
    
 
def get_current_xyz_rotation_gradient(tait_bryan_angles):
    """ 
    Partial derivatives of the xyz rotation matrix at given rotation parameters
    """
    c1, s1 = numpy.cos(tait_bryan_angles[0]), numpy.sin(tait_bryan_angles[0])
    c2, s2 = numpy.cos(tait_bryan_angles[1]), numpy.sin(tait_bryan_angles[1])
    c3, s3 = numpy.cos(tait_bryan_angles[2]), numpy.sin(tait_bryan_angles[2])
    
    d1 = numpy.asarray([[0.0, 0.0, 0.0],
                        [-s1*s3+c3*c1*s2, -s1*c3-c1*s2*s3, -c2*c1],
                        [c1*s3+s1*c3*s2, c3*c1-s1*s2*s3, -s1*c2]])
    
    d2 = numpy.asarray([[-s2*c3, s2*s3, c2],
                        [c3*s1*c2, -s1*c2*s3, s2*s1],
                        [-c1*c3*c2, c1*c2*s3, -c1*s2]])
    
    d3 = numpy.asarray([[-c2*s3, -c2*c3, 0.0],
                        [c1*c3-s3*s1*s2, -c1*s3-s1*s2*c3, 0.0],
                        [s1*c3+c1*s3*s2, -s3*s1+c1*s2*c3, 0.0]])
    
    grad_rot = numpy.vstack((d1, d2, d3))
    
    return grad_rot 
 
    
def generate_transformation_zyx_euler_angles(angle_z, angle_y, angle_x, trans_x, trans_y, trans_z):
    """
    Generate a homogene transformation matrix (ZYX convention) from ZYX Euler angles 
    
    args:
        angles (float): ZYX angles in degree
        trans (float): ZYX translations 
    """
    def calculate_harmonics(angle):
        """ Cut off rounding errors of the cos/sine computations """
        if numpy.allclose(0.0, numpy.sin(angle), 1e-5, 1e-8) == True:
            s = 0.0
        else:
            s = numpy.sin(angle)
        if numpy.allclose(1.0, numpy.sin(angle), 1e-5, 1e-8) == True:
            s = 1.0
        else:
            s = numpy.sin(angle)
        if numpy.allclose(0.0, numpy.cos(angle), 1e-5, 1e-8) == True:
            c = 0.0
        else:
            c = numpy.cos(angle)
        if numpy.allclose(1.0, numpy.cos(angle), 1e-5, 1e-8) == True:
            c = 1.0
        else:
            c = numpy.cos(angle)  
        
        return c, s
    
    angle_x, angle_y, angle_z = numpy.radians(angle_x), numpy.radians(angle_y), numpy.radians(angle_z)
    c1, s1 = calculate_harmonics(angle_z)
    c2, s2 = calculate_harmonics(angle_y)
    c3, s3 = calculate_harmonics(angle_x)
     
    transformation = numpy.asarray([[c1*c2, c1*s2*s3-c3*s1, s1*s3+c1*c3*s2, trans_x],
                                    [c2*s1 ,c1*c3+s1*s2*s3 , c3*s1*s2-c1*s3, trans_y],
                                    [-s2, c2*s3, c2*c3, trans_z],
                                    [0.0, 0.0, 0.0, 1.0]])

    return transformation


def get_zyx_angles_from_matrix(R):
    """ Get parameters (ZYX convention) from transformation matrix """
    sy = math.sqrt(R[0, 0] * R[0, 0] +  R[1, 0] * R[1, 0])
    
    singular = sy < 1e-6
 
    if not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    z, y, x = numpy.degrees(z), numpy.degrees(y), numpy.degrees(x)
    
    return numpy.array([z, y, x])


def distances_histogram(distances):
    """ Show a histogram for a numpy array of distances """
    fig = matplotlib.pyplot.figure()
    matplotlib.pyplot.title("Remaining distances after convergence", fontsize=15)
    matplotlib.pyplot.ylabel("# of points per bin", fontsize=15)
    matplotlib.pyplot.xlabel("Distance to nearest neighbor", fontsize=15)
    num = distances.shape[0]
    matplotlib.pyplot.hist(distances, bins=int(num/100))
    matplotlib.pyplot.show()
    
    return 

    
def mean_squared_error(X, Y):
    """ 
    Compute the euclidean distance between all point pairs and the rms of it
    """
    sq_dist = ((X - Y)**2).sum(axis=1)
    rmse = numpy.sqrt(numpy.mean(sq_dist))
    
    return rmse


def plot_rmse(rmse_list):
    """ Plot the rms error """
    fig = matplotlib.pyplot.figure()
    matplotlib.pyplot.title("RMS error over iterations", fontsize=15)
    matplotlib.pyplot.ylabel("RMSE", fontsize=15)
    matplotlib.pyplot.xlabel("Number of iterations", fontsize=15)
    matplotlib.pyplot.plot(numpy.arange(len(rmse_list)), rmse_list, "-")
    matplotlib.pyplot.show()

    return


    
    
    
