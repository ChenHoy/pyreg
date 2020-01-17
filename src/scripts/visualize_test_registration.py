#!/usr/bin/env python3

"""
Compute matchings of a transformed set via the ICP
"""

import os
import numpy 
import copy
import pyreg.functions
from pyreg.point_cloud_class import *
import ipdb # whenever you use IPython together with mlab: ipython --gui=qt and then run script.py!
import pyreg.visualization


def main():
    """ Short script for visualizing the registration process """
    script_dir = os.getcwd()
    test_dir = script_dir+"/../../../data/robustness/test_2018_09_30/icp_2018_09_29-12_46_47/"
    os.chdir(test_dir+"rot_z_-60/") # Do this manually
    
    target = PointCloud()
    scan = PointCloud()
    try:
        target.read_pcd("input/target.xyz")
        scan.read_pcd("input/scan.xyz")
    except (SystemExit, KeyboardInterrupt):
        raise Exception("Failed to open file")

    # Draw problem
    pyreg.visualization.draw(scan, target)

    # Draw result after registration
    results = pyreg.functions.load_results("output/results")    
    transform_estimate = results["estimated_transformation"]
    scan.transform(transform_estimate)
    pyreg.visualization.draw(scan, target)
    
    # Draw remaining distances with heat map
    deviations = results["remaining_deviations"]
    pyreg.visualization.color_deviations(scan, deviations)


if __name__ == "__main__":
    main()
