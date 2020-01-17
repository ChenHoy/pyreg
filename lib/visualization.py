from mayavi import mlab
import open3d
from pyreg.point_cloud_class import *
import pyreg.functions
import io
import os
import copy


@mlab.show
def draw_stl_surface(stl_filepath):
    """ 
    Draw a stl surface given through a filepath. This is useful when first visualizing a part, without doing anything else.
    """
    engine = Engine()
    fig = mlab.figure(engine = engine)
    surface_data = engine.open(stl_filepath)
    opened_surface = mlab.pipeline.surface(surface_data)
    mlab.pipeline.surface(opened_surface, figure = fig)


@mlab.show
def draw_feature_intensity(pcd, feature_values, cmap="hot"):
    """ 
    Make a colored point cloud with colorcoded feature values.
    
    args:
        pcd (PointCloud):           point cloud object
        feature_values (ndarray):   numpy array of scalar values
        cmap (str):                 color map supported by mayavi      
    """
    fig = mlab.figure(bgcolor=(1,1,1))
    
    pcd_pts = mlab.points3d(pcd.points[:, 0], pcd.points[:, 1], pcd.points[:, 2], feature_values, 
                            colormap=cmap, mode="point")
                            
    cmap = mlab.colorbar(orientation="vertical", nb_labels=5, label_fmt="%.4f")
    cmap.label_text_property.color = (0.0, 0.0, 0.0)
    
    fig.scene.isometric_view()
    fig.scene.parallel_projection = True
    
    return


# TODO Generalize this to arbitrary amount of point clouds
@mlab.show
def compare_curvature(pcd1, pcd2, cmap="jet"):
    """ 
    Compare the curvature of two point clouds
    """
    fig = mlab.figure(bgcolor=(1,1,1))
    
    pcd1_pts = mlab.points3d(pcd1.points[:, 0], pcd1.points[:, 1], pcd1.points[:, 2], pcd1.curvature.flatten(), 
                            colormap=cmap, mode="point")
    
    pcd2_pts = mlab.points3d(pcd2.points[:, 0], pcd2.points[:, 1], pcd2.points[:, 2], pcd2.curvature.flatten(), 
                            colormap=cmap, mode="point")
    
    cmap = mlab.colorbar(orientation="vertical", nb_labels=5, label_fmt="%.4f")
    cmap.label_text_property.color = (0.0, 0.0, 0.0)
    
    fig.scene.isometric_view()
    fig.scene.parallel_projection = True
    
    return


@mlab.show
def draw(pcd, *args):
    """ 
    Draw as many PointCloud objects as given, the color list may need to be updated as it's only for 4 clouds
    """
    fig = mlab.figure(bgcolor=(1,1,1))
    
    color_list = [(1, 0, 0), (0, 1, 0), (1, 0, 1), (1, 1, 0)]
    
    if isinstance(pcd, PointCloud) == True:
        pcd_pts = mlab.points3d(pcd.points[:, 0], pcd.points[:, 1], pcd.points[:, 2], 
                                color=(0, 0, 1), mode="point")    
    elif isinstance(pcd, numpy.ndarray) == True:
        pcd_pts = mlab.points3d(pcd[:, 0], pcd[:, 1], pcd[:, 2], 
                                color=(0, 0, 1), mode="point")    
    else:
        raise TypeError("Point clouds need to be of type (ndarray) or (PointCloud)!")
        
    for idx, arg in enumerate(args):
        if isinstance(arg, PointCloud) == True:
            idx = mlab.points3d(arg.points[:, 0], arg.points[:, 1], arg.points[:, 2], 
                                    color=color_list[idx], mode="point")    
        elif isinstance(pcd_temp, numpy.ndarray) == True:
            idx = mlab.points3d(arg[:, 0], arg[:, 1], arg[:, 2], 
                                    color=color_list[idx], mode="point")
        else:
            raise TypeError("Point clouds need to be of type (ndarray) or (PointCloud)!")
    
    fig.scene.isometric_view()
    fig.scene.parallel_projection = True

    return


@mlab.show
def draw_correspondences(scan, target, correspondence_set):
    """ 
    Draw a line between corresponding points of two sets 
    
    args:
        scan (ndarray):                 Scan point cloud
        target (ndarray):               Target point cloud
        correspondence_set (ndarray):   Correspondences between scan and target. 1. col is scan point idx, 2. col is target point idx
    """
    scan_temp = copy.deepcopy(scan.points)    
    target_temp = copy.deepcopy(target.points)
    fig = mlab.figure(bgcolor=(1,1,1))
    
    scan_pts = mlab.points3d(scan_temp[:, 0], scan_temp[:, 1], scan_temp[:, 2], 
                                color=(1, 0, 0), mode="point", scale_factor=2.0)
    target_pts = mlab.points3d(target_temp[:, 0], target_temp[:, 1], target_temp[:, 2], 
                                color=(0, 0, 1), mode="point", scale_factor=2.0)
    
    # We merge all points to one set, where each point p_i corresponds to q_(N+i) in order to use dataset.lines and surface plot
    merged_points = numpy.vstack((scan_temp[correspondence_set[:, 0]], target_temp[correspondence_set[:, 1]]))
    connections = []
    N = correspondence_set.shape[0]
    for i in range(N):
        connections.append([i, N+i])
    
    points = mlab.points3d(merged_points[:, 0], merged_points[:, 1], merged_points[:, 2], 
                                color=(0, 0, 0), mode="point")
    points.mlab_source.dataset.lines = connections
    points.mlab_source.reset()
    
    mlab.pipeline.surface(points, color=(0, 0, 0),
                              representation='wireframe',
                              line_width=0.5,
                              name='Connections')

    
    fig.scene.isometric_view()
    fig.scene.parallel_projection = True
    
    return
    

@mlab.show
def draw_registration(scan, target):
    """ 
    Draws scan (red) and target/model (green) in seperate window using mayavi visualizer. This will be used for pairwise alignment. 
    """
    scan_temp = copy.deepcopy(scan)    
    target_temp = copy.deepcopy(target)
    fig = mlab.figure(bgcolor=(1,1,1))
    
    if isinstance(scan_temp, PointCloud) == True:
        scan_pts = mlab.points3d(scan_temp.points[:, 0], scan_temp.points[:, 1], scan_temp.points[:, 2], 
                                color=(1, 0, 0), mode="point")    
    elif isinstance(scan_temp, numpy.ndarray) == True:
        scan_pts = mlab.points3d(scan_temp[:, 0], scan_temp[:, 1], scan_temp[:, 2], 
                                color=(1, 0, 0), mode="point")    
    else:
        raise TypeError("Point clouds need to be of type (ndarray) or (PointCloud)!")

    if isinstance(target_temp, PointCloud) == True:
        target_pts = mlab.points3d(target_temp.points[:, 0], target_temp.points[:, 1], target_temp.points[:, 2], 
                                color=(0, 0, 1), mode="point")    
    elif isinstance(target_temp, numpy.ndarray) == True:
        target_pts = mlab.points3d(target_temp[:, 0], target_temp[:, 1], target_temp[:, 2], 
                                color=(0, 0, 1), mode="point")
    else:
        raise TypeError("Point clouds need to be of type (ndarray) or (PointCloud)!")
    
    
    fig.scene.isometric_view()
    fig.scene.parallel_projection = True

    return


@mlab.show
def draw_key_points(pcd):
    """ 
    Draw a point cloud together with its key points. 
    
    args:
        pcd (PointCloud):       point cloud in PointCloud format. The key points need to be computed before.
    """
    if pcd.has_key_points() == False:
        raise Exception("Point cloud has no key points!")
    else:
        mean_nn = pyreg.functions.get_mean_nn_distance(pcd)
        fig = mlab.figure(bgcolor=(1,1,1))
        pcd_pts = mlab.points3d(pcd.points[:, 0], pcd.points[:, 1], pcd.points[:, 2], 
                                color=(0, 0, 1), mode="point")
        pcd_key_pts = mlab.points3d(pcd.key_points[0][:, 0], pcd.key_points[0][:, 1], pcd.key_points[0][:, 2], 
                                color=(1, 0, 0), mode="sphere", scale_factor=mean_nn)
    fig.scene.isometric_view()
    fig.scene.parallel_projection = True
    
    return
    

@mlab.show    
def compare_key_points(pcd1, pcd2):
    """
    Compare the key points from two point clouds. Ideally, very similar points are detected independent from the view point.
    """
    if pcd1.has_key_points() == False or pcd2.has_key_points() == False:
        raise Exception("Point clouds have no key points!")
    else:
        mean_nn1 = pyreg.functions.get_mean_nn_distance(pcd1)
        mean_nn2 = pyreg.functions.get_mean_nn_distance(pcd2)
        fig = mlab.figure(bgcolor=(1,1,1))
        pcd1_pts = mlab.points3d(pcd1.points[:, 0], pcd1.points[:, 1], pcd1.points[:, 2], 
                                color=(1, 0, 0), mode="point")
        pcd1_key_pts = mlab.points3d(pcd1.key_points[0][:, 0], pcd1.key_points[0][:, 1], pcd1.key_points[0][:, 2], 
                                color=(0, 1, 0), mode="sphere", scale_factor=mean_nn1)
                                
        
        pcd2_pts = mlab.points3d(pcd2.points[:, 0], pcd2.points[:, 1], pcd2.points[:, 2], 
                                color=(0, 0, 1), mode="point")
        pcd2_key_pts = mlab.points3d(pcd2.key_points[0][:, 0], pcd2.key_points[0][:, 1], pcd2.key_points[0][:, 2], 
                                color=(1, 0, 1), mode="sphere", scale_factor=mean_nn2)
                                
    fig.scene.isometric_view()
    fig.scene.parallel_projection = True
    
    
    return
    

def save_fig(scan, target, filepath):
    """ 
    Save the figure of pairwise registration in isometric view with parallel projection.  
    """
    scan_temp = copy.deepcopy(scan)    
    target_temp = copy.deepcopy(target)
    
    mlab.options.offscreen = True
    fig = mlab.figure(bgcolor=(1,1,1))
    
    if isinstance(scan_temp, PointCloud) == True:
        scan_pts = mlab.points3d(scan_temp.points[:, 0], scan_temp.points[:, 1], scan_temp.points[:, 2], 
                                color=(1, 0, 0), mode="point")    
    elif isinstance(scan_temp, numpy.ndarray) == True:
        scan_pts = mlab.points3d(scan_temp[:, 0], scan_temp[:, 1], scan_temp[:, 2], 
                                color=(1, 0, 0), mode="point")    
    else:
        raise TypeError("Point clouds need to be of type (ndarray) or (PointCloud)!")

    if isinstance(target_temp, PointCloud) == True:
        target_pts = mlab.points3d(target_temp.points[:, 0], target_temp.points[:, 1], target_temp.points[:, 2], 
                                color=(0, 0, 1), mode="point")    
    elif isinstance(target_temp, numpy.ndarray) == True:
        target_pts = mlab.points3d(target_temp[:, 0], target_temp[:, 1], target_temp[:, 2], 
                                color=(0, 0, 1), mode="point")
    else:
        raise TypeError("Point clouds need to be of type (ndarray) or (PointCloud)!")
    fig.scene.isometric_view()
    fig.scene.parallel_projection = True
    
    mlab.savefig(filepath, figure=fig)
    mlab.close(all=True)

    return


# TODO Make this visually more appealing
@mlab.show
def plot_normals(pcd):
    """ 
    Plot the point cloud together with computed normals using mayavi (VTK).
    
    args:
        pcd (PointCloud):           point cloud in PointCloud format. Needs precomputed normals. 
    """
    if pcd.has_normals == False:
        raise Exception("This PointCloud object has no normals!")
    else:            
        fig1 = mlab.figure(bgcolor=(1,1,1))
        pcd_pts = mlab.points3d(pcd.points[:, 0], pcd.points[:, 1], pcd.points[:, 2], 
                                    color=(0, 0, 1), mode="point", figure=fig1)
        normals = mlab.quiver3d(pcd.points[:, 0], pcd.points[:, 1], pcd.points[:, 2], 
                                pcd.normals[:, 0], pcd.normals[:, 1], pcd.normals[:, 2],
                                color=(1, 0, 0), figure=fig1, reset_zoom=False)
        mlab.outline()
        
        return


@mlab.show
def color_deviations(pcd, distances, colormap="hot", scale_factor=None):
    """ 
    Colorize the point cloud according to distances with heat map using mayavi visualizer (VTK)
    
    args:
        pcd(PointCloud): Currently only this data type supported
        distances(numpy.array): Distances to each point
    """
    mlab.figure(bgcolor=(1,1,1))
    
    mean_nn = pyreg.functions.get_mean_nn_distance(pcd)
    if scale_factor == None:
        result_pts = mlab.points3d(pcd.points[:, 0], pcd.points[:, 1], pcd.points[:, 2], distances, 
                                    colormap=colormap, mode="point")
    else:
        result_pts = mlab.points3d(pcd.points[:, 0], pcd.points[:, 1], pcd.points[:, 2], distances, 
                                colormap=colormap, mode="sphere", scale_factor=scale_factor)
    cmap = mlab.colorbar(orientation="vertical",nb_labels=3)
    cmap.label_text_property.color = (0.0, 0.0, 0.0)
    mlab.show()
    
    return


# OLD
def color_deviations_open3d(config, result, distances):
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


# OLD
def draw_open3d(source, target, transformation):
    """ Draws the registration result in seperate window with open3d visualizer (GDAL) """
    source_temp = copy.deepcopy(source)    
    target_temp = copy.deepcopy(target)
    
    if isinstance(source_temp, PointCloud) == True:
        pcd_source = open3d.PointCloud()
        pcd_source.points = open3d.Vector3dVector(source_temp.points)
    elif isinstance(source_temp, numpy.ndarray) == True:
        pcd_source = open3d.PointCloud()
        pcd_source.points = open3d.Vector3dVector(source_temp)
    elif isinstance(source_temp, open3d.PointCloud) == True:
        pcd_source = copy.deepcopy(source_temp)
    else:
        raise TypeError("Point clouds need to be of type (ndarray), (PointCloud) or (open3d.PointCloud)!")

    if isinstance(target_temp, PointCloud) == True:
        pcd_target = open3d.PointCloud()
        pcd_target.points = open3d.Vector3dVector(target_temp.points)
    elif isinstance(target_temp, numpy.ndarray) == True:
        pcd_target = open3d.PointCloud()
        pcd_target.points = open3d.Vector3dVector(target_temp)
    elif isinstance(target_temp, open3d.PointCloud) == True:
        pcd_target = copy.deepcopy(target_temp)
    else:
        raise TypeError("Point clouds need to be of type (ndarray) or (PointCloud)!")
    
    # Paint with uniform color
    pcd_source.paint_uniform_color([1, 0.706, 0])
    pcd_target.paint_uniform_color([0, 0.651, 0.929])
    pcd_source.transform(transformation)
    open3d.draw_geometries([pcd_source, pcd_target])

    return
