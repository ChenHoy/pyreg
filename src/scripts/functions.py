import yaml
import os
import numpy 
import open3d
import copy
from contextlib import contextmanager
import logging
from logging.handlers import TimedRotatingFileHandler
from point_cloud_class import *

def create_timed_rotating_log(path):
    logger = logging.getLogger("ICP.log")
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


def read_config():
    """ Read in the yaml config file """
    with open("config.yml", "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
    
    return config


def make_dir(path):
    """ If directory does not exist, create one """
    if not os.path.exists(path):
        os.makedirs(path)


def read_pcd_xyz(script_dir, input_directory, file_name):
    """ Read in a point cloud from a .xyz file """
    abs_file_path = os.path.join(script_dir, input_directory)
    try:
        os.chdir(abs_file_path)
        data = numpy.loadtxt(file_name)
    finally:
        os.chdir(script_dir) # Switch! back to script dir

    return data


def center_open3d(pcd):
    """ Center an open3d point cloud object around the coordinate origin """
    tuple_mean, tuple_covariance = open3d.compute_point_cloud_mean_and_covariance(pcd)
    center_trans = numpy.asarray([[1.0, 0.0, 0.0, -tuple_mean[0]],
                                  [0.0, 1.0, 0.0, -tuple_mean[1]],
                                  [0.0, 0.0, 1.0, -tuple_mean[2]],
                                  [0.0, 0.0, 0.0, 1.0]])
    pcd = pcd.transform(center_trans)
    
    return


def generate_transformation(angle_x, angle_y, angle_z, trans_x, trans_y, trans_z):
    """
    Generate a homogene transformation matrix (XYZ convention) from more readable information 
    
    args:
        angles (float): XYZ angles in degree
        trans (float): XYZ translations 
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
    c1, s1 = calculate_harmonics(angle_x)
    c2, s2 = calculate_harmonics(angle_y)
    c3, s3 = calculate_harmonics(angle_z)
     
    transformation = numpy.asarray([[c2*c3, -c2*s3, s2, trans_x],
                                    [c1*s3+c3*s1*s2, c1*c3-s1*s2*s3, -c2*s1, trans_y],
                                    [s1*s3-c1*c3*s2, c3*s1+c1*s2*s3, c1*c2, trans_z],
                                    [0.0, 0.0, 0.0, 1.0]])

    return transformation


def color_deviations(config, result, distances):
    """ Use the Open3d visualization function to colorize deviations after convergence """
    if isinstance(result, PointCloud) == True:
        result = open3d.PointCloud()
        result.points = open3d.Vector3dVector(result.points)
    elif isinstance(result, numpy.ndarray) == True:
        result = open3d.PointCloud()
        result.points = open3d.Vector3dVector(result)
    elif isinstance(result, open3d.PointCloud) == True:
        pass
    else:
        raise TypeError("Point clouds need to be of type (ndarray), (PointCloud) or (open3d.PointCloud)!")
    
    # Needed Paths
    OPEN3D_BUILD_PATH = "/home/chris/Documents/Masterarbeit/Code/virt_env/env/lib/python3.6/site-packages/Open3D/build/"
    OPEN3D_EXPERIMENTAL_BIN_PATH = OPEN3D_BUILD_PATH + "/bin/Experimental/"
    # TODO use os.findpath or something to make this dynamic
    DATASET_DIR = "/home/chris/Documents/Masterarbeit/Code/virt_env/env/ICP/src/bin"
    
    # Log files
    MY_LOG_POSTFIX = "_COLMAP_SfM.log"
    MY_RECONSTRUCTION_POSTFIX = "_COLMAP.ply"
    
    # File structure
    mvs_outpath = DATASET_DIR + config["colorize_deviations"]["output_path"]
    "/../../data/set1/output/evaluation/"
    scene = config["colorize_deviations"]["scene"] + "/"
    
    # Make directory
    make_dir(mvs_outpath)
    make_dir(mvs_outpath+"/"+scene)
    
    # New log files
    new_logfile = DATASET_DIR + MY_LOG_POSTFIX
    colmap_ref_logfile = DATASET_DIR + scene + "_COLMAP_SfM.log"
    mvs_file = mvs_outpath + scene + MY_RECONSTRUCTION_POSTFIX
    
    # Write the distances to bin files
    numpy.array(distances).astype("float64").tofile(mvs_outpath + "/" + scene + ".precision.bin")
    
    # Colorize the poincloud files with the precision and recall values
    open3d.write_point_cloud(mvs_outpath + "/" + scene + ".precision.ply", result)
    open3d.write_point_cloud(mvs_outpath + "/" + scene + ".precision.ncb.ply", result)
    result_n_fn = mvs_outpath + "/" + scene + ".precision.ply"
    
    eval_str_viewDT = OPEN3D_EXPERIMENTAL_BIN_PATH + "ViewDistances " + result_n_fn + " --max_distance " + str(numpy.asarray(distances).max()) + " --write_color_back --without_gui"
    os.system(eval_str_viewDT)
    
    result_colored = open3d.read_point_cloud(result_n_fn)
    open3d.draw_geometries([result_colored])

    return


def plot_rmse(iterations, rmse_array):
    """ Plot the rms error """
    fig = matplotlib.pyplot.figure()
    matplotlib.pyplot.title("RMS error over iterations")
    matplotlib.pyplot.xlabel("Iterations")
    matplotlib.pyplot.ylabel("not actual RMS error")
    
    rmse = rmse_array[0:iterations]
    matplotlib.pyplot.plot(numpy.arange(iterations), rmse, "-")

    return


def draw_registration_result(source, target, transformation):
    """ Draws the registration result in seperate window """
    source_temp = copy.deepcopy(source)    
    target_temp = copy.deepcopy(target)
    
    if isinstance(source, PointCloud) == True:
        pcd_source = open3d.PointCloud()
        pcd_source.points = open3d.Vector3dVector(source.points)
    elif isinstance(source, numpy.ndarray) == True:
        pcd_source = open3d.PointCloud()
        pcd_source.points = open3d.Vector3dVector(source)
    elif isinstance(source, open3d.PointCloud) == True:
        pcd_source = copy.deepcopy(source)
    else:
        raise TypeError("Point clouds need to be of type (ndarray) or (PointCloud)!")

    if isinstance(target, PointCloud) == True:
        pcd_target = open3d.PointCloud()
        pcd_target.points = open3d.Vector3dVector(target.points)
    elif isinstance(target, numpy.ndarray) == True:
        pcd_target = open3d.PointCloud()
        pcd_target.points = open3d.Vector3dVector(target)
    elif isinstance(target, open3d.PointCloud) == True:
        pcd_target = copy.deepcopy(target)
    else:
        raise TypeError("Point clouds need to be of type (ndarray) or (PointCloud)!")
    
    # Paint with uniform color
    pcd_source.paint_uniform_color([1, 0.706, 0])
    pcd_target.paint_uniform_color([0, 0.651, 0.929])
    pcd_source.transform(transformation)
    open3d.draw_geometries([pcd_source, pcd_target])

    return
    


