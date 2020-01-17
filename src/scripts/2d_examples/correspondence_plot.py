#!/usr/bin/env python3

"""
Illustrate the algorithms with simple 2D shapes
"""

import os
import copy
import numpy
import matplotlib
#~ matplotlib.use("pgf") 
#~ matplotlib.rcParams["mathtext.default"] = "regular"
#~ preamble = {
    #~ "font.family": "lmodern",  # use this font for text elements
    #~ "text.usetex": True, # use inline math for ticks
    #~ "pgf.rcfonts": False, # don"t setup fonts from rc parameters
    #~ "pgf.texsystem": "xelatex",
    #~ "pgf.preamble": [
        #~ "\\usepackage{amsmath}",
        #~ "\\usepackage{amssymb}",
        #~ "\\usepackage{lmodern}",
        #~ ]
    #~ }
#~ matplotlib.rcParams.update(preamble)
import tools
import scipy
import matplotlib.gridspec as gridspec
from sklearn import mixture
from pyreg.icpy.icp_registration import IcpRegistration
from pyreg.cpyd.cpd_rigid_registration import CpdRigidRegistration
from pyreg.registration_base import ConvergenceCriteria
import pyreg.functions
from pyreg.point_cloud_class import *
import ipdb 


def plot_icp_reg(model, tscan, **kwargs):
    """ Show the registered scan and correspondences """
    fig, ax = matplotlib.pyplot.subplots()
    ax.scatter(tscan.points[:, 0], tscan.points[:, 1], color="r", label="scan")
    ax.scatter(model.points[:, 0], model.points[:, 1], color="b", label="model")
    
    for key, value in kwargs.items():
        if key == "correspondences":        
            for correspondence in value:
                ax.plot([tscan.points[correspondence[0], 0], model.points[correspondence[1], 0]], 
                     [tscan.points[correspondence[0], 1], model.points[correspondence[1], 1]], 
                     "--", color="k")
        else:
            pass
    
    ax.set_ylabel("y coordinate", fontsize=15)
    ax.set_xlabel("x coordinate", fontsize=15)
    ax.legend(loc="upper right")
    ax.set_autoscale_on
    matplotlib.pyplot.show()
    

def plot_cpd_reg(model, tscan, correspondences):
    """ Show the registered scan and correspondences """
    fig, ax = matplotlib.pyplot.subplots()
    ax.scatter(tscan.points[:, 0], tscan.points[:, 1], color="r", label="scan")
    ax.scatter(model.points[:, 0], model.points[:, 1], color="b", label="model")
    
    cmap = matplotlib.pyplot.cm.get_cmap("YlOrRd", 20)
    sm = matplotlib.pyplot.cm.ScalarMappable(norm=None, cmap=cmap)
    sm.set_array([])
    
    for i in range(correspondences.shape[0]): # rows
        for j in range(correspondences.shape[1]): # cols
            ax.plot([tscan.points[i, 0], model.points[j, 0]], 
                    [tscan.points[i, 1], model.points[j, 1]], 
                    "--", color=cmap(correspondences[i][j]))
    
    fig.colorbar(sm, ax=ax)
    
    ax.set_ylabel("y coordinate", fontsize=15)
    ax.set_xlabel("x coordinate", fontsize=15)
    ax.legend(loc="upper right")
    
    matplotlib.pyplot.show()
    

def plot_gmm_log_likelihood(centroids, data_observations):
    """ GMM plot of Model points to for illustration """
    clf = mixture.GaussianMixture(n_components=centroids.points.shape[0], covariance_type='full')
    clf.fit(centroids.points)
    
    x_range = (numpy.amax(centroids.points[:, 0])+0.3)-(numpy.amin(centroids.points[:, 0])-0.3)
    y_range = (numpy.amax(centroids.points[:, 1])+0.3)-(numpy.amin(centroids.points[:, 1])-0.3)
    
    if y_range > x_range:
        delta_range = y_range - x_range
        y = numpy.linspace(numpy.amin(centroids.points[:, 1])-0.3, numpy.amax(centroids.points[:, 1])+0.3)
        x = numpy.linspace(numpy.amin(centroids.points[:, 0])-delta_range/2-0.3, numpy.amax(centroids.points[:, 0])+delta_range/2+0.3)
    else:
        delta_range = x_range - y_range
        y = numpy.linspace(numpy.amin(centroids.points[:, 1])-0.3-delta_range/2, numpy.amax(centroids.points[:, 1])+0.3+delta_range/2)
        x = numpy.linspace(numpy.amin(centroids.points[:, 0])-0.3, numpy.amax(centroids.points[:, 0])+0.3)
    
    #~ x = numpy.linspace(numpy.amin(centroids.points[:, 0])-0.2, numpy.amax(centroids.points[:, 0])+0.2)
    #~ y = numpy.linspace(numpy.amin(centroids.points[:, 1])-0.2, numpy.amax(centroids.points[:, 1])+0.2)
    X, Y = numpy.meshgrid(x, y)
    XX = numpy.array([X.ravel(), Y.ravel()]).T
    Z = -clf.score_samples(XX)
    Z = Z.reshape(X.shape)

    fig = matplotlib.pyplot.figure()
    #~ lvls = numpy.logspace(0, 5, 20)
    cmap = matplotlib.pyplot.cm.get_cmap("viridis")
    CS = matplotlib.pyplot.contourf(X, Y, Z, cmap=cmap, norm=matplotlib.colors.LogNorm(), alpha=0.5)
    matplotlib.pyplot.contour(X, Y, Z, alpha=0.5)
    matplotlib.pyplot.colorbar(CS)  
    #~ CB = matplotlib.pyplot.colorbar(CS, norm=, levels=lvls)
    matplotlib.pyplot.scatter(centroids.points[:, 0], centroids.points[:, 1], color="r", label="centroids")

    matplotlib.pyplot.legend(loc="upper right")
    fig.savefig("gmm.pdf", bbox_inches="tight")
    matplotlib.pyplot.close("all")

    return


def main():
    script_dir = os.getcwd()
    config = pyreg.functions.read_config("config.yml")
    log_file = "registration.log"
    logger = pyreg.functions.create_timed_rotating_log(log_file)
    # Polygon
    #~ polygon = tools.gen_ran_poly(10000, 2)
    
    model = PointCloud(points=numpy.array([[0.5, 0.0], [1.0, 0.5], [1.5, 1.5], [1.0, 2.5]]))
    model.center_around_origin()
    scan = PointCloud(points=numpy.array([[0.5, 0.0], [1.0, 0.5], [1.5, 1.5], [1.0, 2.5]]))
    scan.center_around_origin()
    
    transformation = numpy.eye(3)
    rotation = tools.rotation_two(numpy.radians(40.0))
    translation = numpy.array([2.0, 0])
    transformation[0:2, 0:2] = rotation
    transformation[:-1, 2] = translation
    
    scan.transform(transformation)
    #~ plot_icp_reg(model, scan)
    plot_gmm_log_likelihood(scan, model)
    
    tscan = copy.deepcopy(scan)
    
    max_iteration = 10
    num_points = scan.points.shape[0]
    points_iter = numpy.zeros((max_iteration+1, num_points, 2))
    points_iter[0] = scan.points
    
    estimate = numpy.identity(3)
    for i in range(max_iteration):
        trans_init = numpy.identity(3)
        icp_reg = IcpRegistration(copy.deepcopy(tscan), copy.deepcopy(model), trans_init, config, logger, 
                                ConvergenceCriteria(max_iteration=1,
                                                    tolerance=1e-6,
                                                    error_metric="point_to_point")
                                )
                                
        #~ cpd_reg = CpdRigidRegistration(copy.deepcopy(tscan), copy.deepcopy(model), trans_init, config, logger, 
                                   #~ ConvergenceCriteria(max_iteration=1,
                                                    #~ tolerance=1e-6),
                                    #~ w=1-model.points.shape[0]/tscan.points.shape[0]
                                    #~ )
        
        #~ cpd_reg.register()
        #~ estimate = cpd_reg.transformation @ estimate 
        #~ tscan.transform(cpd_reg.transformation)
        icp_reg._estimation_step()
        plot_icp_reg(model, tscan, correspondences=icp_reg.correspondence_set)
        
        #~ plot_icp_reg(model, tscan)
        #~ cpd_reg._e_step()
        #~ plot_cpd_reg(model, tscan, correspondences=cpd_reg.P)


if __name__ == "__main__":
    main()    
