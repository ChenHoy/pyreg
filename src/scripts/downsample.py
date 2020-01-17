#!/usr/bin/env python3

"""
Downsample the data set
"""

import os
import open3d
import numpy
import pyreg.functions
from pyreg.point_cloud_class import *
import ipdb 
from mayavi import mlab
import copy


def main():
    script_dir = os.getcwd()
    config = pyreg.functions.read_config()
    
    bunny_raw = PointCloud()
    
    try:
        bunny_raw.read_pcd(config["files"]["source"])
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception:
        raise Exception("File not found!")
    
    bunny_raw.center_around_origin()
    
    bunny_raw_open3d = open3d.PointCloud()
    bunny_raw_open3d.points = open3d.Vector3dVector(bunny_raw.points)
    
    bunny_open3d = open3d.voxel_down_sample(bunny_raw_open3d, voxel_size = config["downsample"]["voxel_size"])
    bunny = PointCloud(points=copy.deepcopy(numpy.asarray(bunny_open3d.points)))
    
    mlab.figure()
    bunny_pts = mlab.points3d(bunny.points[:, 0], bunny.points[:, 1], bunny.points[:, 2], 
                                color=(0, 1, 0), mode="point")
    mlab.outline()
    
    ipdb.set_trace()
    
    bunny.write_pcd(file_format="txt", file_path=config["files"]["output"]+"bunny_7000.xyz")
    
    
if __name__ == "__main__":
    main()

