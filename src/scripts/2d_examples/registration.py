#!/usr/bin/env python3

"""
Illustrate the algorithms with simple 2D shapes
"""

import os
import copy
import numpy
import matplotlib.pyplot
import tools
import scipy
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from sklearn import mixture
from pyreg.icpy.icp_registration import IcpRegistration
from pyreg.cpyd.cpd_rigid_registration import CpdRigidRegistration
from pyreg.registration_base import ConvergenceCriteria
import pyreg.functions
from pyreg.point_cloud_class import *
import ipdb 


def show_approx_motion(target, scan, trace_scan):
    """ Shows the approximated trace of the points during a transformation """
    fig = matplotlib.pyplot.figure()
    matplotlib.pyplot.title("Approximated point motion")
    matplotlib.pyplot.xlabel("x coordinates")
    matplotlib.pyplot.ylabel("y coordinates")
    
    all_x = trace_scan.T[0] # N x i matrix of all x values
    all_y = trace_scan.T[1] # N x i matrix of all y values
    num_points = all_x.shape[0]
    
    for point in range(num_points):
        f = scipy.interpolate.interp1d(all_x[point], all_y[point], kind="slinear")
        xnew = numpy.linspace(numpy.amin(all_x[point]), numpy.amax(all_x[point]), num=50)
        ynew = f(xnew)
        matplotlib.pyplot.plot(xnew, ynew, "r--")
        
    matplotlib.pyplot.scatter(scan.points[:, 0], scan.points[:, 1], color="r", label="Scan")
    matplotlib.pyplot.scatter(target.points[:, 0], target.points[:, 1], color="b", label="Target")
    matplotlib.pyplot.legend(loc="upper right")
    matplotlib.pyplot.show()


def plot_poly_reg(target, tscan, **kwargs):
    """ Show the registered scan and correspondences """
    fig = matplotlib.pyplot.figure()
    matplotlib.pyplot.title("Registration of a Polygon")
    matplotlib.pyplot.scatter(tscan.points[:, 0], tscan.points[:, 1], color="r", label="Scan")
    matplotlib.pyplot.scatter(target.points[:, 0], target.points[:, 1], color="b", label="Target")
    
    for key, value in kwargs.items():
        if key == "correspondences":        
            for correspondence in value:
                matplotlib.pyplot.plot([tscan.points[correspondence[0], 0], target.points[correspondence[1], 0]], 
                     [tscan.points[correspondence[0], 1], target.points[correspondence[1], 1]], 
                     "--", color="k")
        else:
            pass
    
    matplotlib.pyplot.ylabel("y", fontsize=15)
    matplotlib.pyplot.xlabel("x", fontsize=15)
    matplotlib.pyplot.legend(loc="upper right")

    matplotlib.pyplot.show()
    

def plot_gmm_log_likelihood(data_set):
    """ GMM plot of Model points to for illustration """
    clf = mixture.GaussianMixture(n_components=data_set.points.shape[0], covariance_type='full')
    clf.fit(data_set.points)
    
    x = numpy.linspace(numpy.amin(data_set.points[:, 0])-0.2, numpy.amax(data_set.points[:, 0])+0.2)
    y = numpy.linspace(numpy.amin(data_set.points[:, 1])-0.2, numpy.amax(data_set.points[:, 1])+0.2)
    X, Y = numpy.meshgrid(x, y)
    XX = numpy.array([X.ravel(), Y.ravel()]).T
    Z = -clf.score_samples(XX)
    Z = Z.reshape(X.shape)

    fig = matplotlib.pyplot.figure()
    lvls = numpy.logspace(0, 5, 20)
    CS = matplotlib.pyplot.contour(X, Y, Z, norm=LogNorm(), levels=lvls)
    CB = matplotlib.pyplot.colorbar(CS, ticks=lvls)
    matplotlib.pyplot.scatter(data_set.points[:, 0], data_set.points[:, 1], color="r")

    matplotlib.pyplot.title('Negative log-likelihood predicted by a GMM')
    matplotlib.pyplot.axis('tight')
    matplotlib.pyplot.show()

    return


def manual_SVD(X, Y):
    """ We want to know if it is possible to register even large pertubations as long correspondence estimates are good """
    X_mean = numpy.mean(X, axis=0)
    Y_mean = numpy.mean(Y, axis=0)
    XX = X - X_mean
    YY = Y - Y_mean
    
    H = XX.T @ YY     
    U, S, Vt = numpy.linalg.svd(H)
    R = Vt.T @ U.T
    
    if numpy.linalg.det(R) < 0: # special reflection case
        Vt[self.D-1, :] *= -1
        R = Vt.T @ U.T

    t = Y_mean.T - R @ X_mean.T
    
    D = X.shape[1]
    T = numpy.eye(D+1)
    T[:D, :D] = R
    T[:D, D] = t 

    return T


def main():
    script_dir = os.getcwd()
    config = pyreg.functions.read_config("config.yml")
    log_file = "registration.log"
    logger = pyreg.functions.create_timed_rotating_log(log_file)

    polygon = tools.gen_ran_poly(10, 2)
    
    target = PointCloud(points=polygon)
    target.center_around_origin()
    scan = PointCloud(points=polygon)
    scan.center_around_origin()
    
    #~ plot_gmm_log_likelihood(target)
    
    transformation = numpy.eye(3)
    rotation = tools.rotation_two(numpy.radians(-180.0))
    translation = numpy.array([0, 0])
    transformation[0:2, 0:2] = rotation
    transformation[:-1, 2] = translation
    
    scan.transform(transformation)
    tscan = copy.deepcopy(scan)
    
    max_iteration = 5
    num_points = scan.points.shape[0]
    all_points = numpy.zeros((max_iteration+1, num_points, 2))
    all_points[0] = scan.points

    #~ plot_gmm_log_likelihood(scan)
    
    estimate = numpy.identity(3)
    
    for i in range(max_iteration):
        trans_init = numpy.identity(3)
        #~ reg = IcpRegistration(copy.deepcopy(tscan), copy.deepcopy(target), trans_init, config, logger, 
                                #~ ConvergenceCriteria(max_iteration=1,
                                                    #~ tolerance=1e-6,
                                                    #~ error_metric="point_to_point"))
        reg = CpdRigidRegistration(copy.deepcopy(tscan), copy.deepcopy(target), trans_init, config, logger, 
                                   ConvergenceCriteria(max_iteration=1,
                                                    tolerance=1e-6)
                                    )
        
        reg.register()
        estimate = reg.transformation @ estimate 
        
        tscan.transform(reg.transformation)
        all_points[i+1] = tscan.points
        #~ plot_poly_reg(target, tscan, "correspondences"=reg.correspondence_set)
        plot_poly_reg(target, tscan)
    
    #~ show_approx_motion(target, scan, all_points)
 
    

if __name__ == "__main__":
    main()    
