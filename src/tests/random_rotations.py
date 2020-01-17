#!/usr/bin/env python3

import numpy
import matplotlib.pyplot
from mpl_toolkits.mplot3d import axes3d
import scipy.linalg
import scipy.interpolate 
import ipdb

    
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


def main():
    """ Generate random rotations in SO(3) """
    v = numpy.ones((3, 1))
    norm = numpy.linalg.norm(v)
    v = v/norm
    origin = numpy.zeros((3, 1))
    
    R_euler = generate_random_rotation_euler()
    R_quat = generate_random_rotation_quaternion()
    
    n = 1000
    v_rot_quat = numpy.zeros((n, 3, 1))
    v_rot_euler = numpy.zeros((n, 3, 1))
    for i in range(n):
        R_quat = generate_random_rotation_quaternion()
        v_rot_quat[i] = R_quat @ v
        R_euler = generate_random_rotation_euler()
        v_rot_euler[i] = R_euler @ v
        
    fig = matplotlib.pyplot.figure()
    ax = fig.gca(projection="3d")
    ax.scatter(v_rot_quat[:, 0], v_rot_quat[:, 1], v_rot_quat[:, 2], label="Quaternion")
    matplotlib.pyplot.legend(loc="upper right")
    
    fig = matplotlib.pyplot.figure()
    ax = fig.gca(projection="3d")
    ax.scatter(v_rot_euler[:, 0], v_rot_euler[:, 1], v_rot_euler[:, 2], label="Axis/angle")
    
    matplotlib.pyplot.legend(loc="upper right")
    matplotlib.pyplot.show()
    

if __name__ == "__main__":
    main()
