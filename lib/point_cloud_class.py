import numpy
from sklearn.neighbors import NearestNeighbors
import bottleneck
from functools import reduce
from sklearn import preprocessing
from plyfile import PlyData, PlyElement
import copy
import open3d
import scipy.io
import csv
from stl import mesh
import ipdb
import os
import time


class PointCloud(object):
    """ 
    Point cloud class
    
    args: 
        points (ndarray): points 
        normals (ndarray): estimated surface normal of a point
        curvature (ndarray): estimated curfature value of a point
        color (ndarray): color information of a point 
        fpfh (nx33 ndarray): FastPointFeatureHistogram for each point
        key_points (nx3 ndarray, nx1 ndarry): Key points that are identified from a feature. Points in coords and index are saved
    """    
    def __init__(self, points=None, normals=None, color=None, curvature=None, bounding_box=None, fpfh=None, key_points=None):
        """ Initialize Point cloud object """
        if points is not None and points == []: 
            raise Exception("Input points are empty!")
        else:
            pass

        self.points = None if points is None else points
        self.normals = None if normals is None else normals 
        self.color = None if color is None else color
        self.curvature = None if curvature is None else curvature
        self.bounding_box = None if bounding_box is None else bounding_box
        self.fpfh = None if fpfh is None else fpfh
        self.key_points = None if key_points is None else key_points
    
    
    def read_pcd(self, file_path):
        """ Read in a point cloud from to-be-specified file format into numpy array """
        file_name, file_extension = os.path.splitext(file_path)
        
        if file_extension == ".yml" or file_extension == ".yaml":
            stream = io.open(filepath, "r")
            data_loaded = yaml.safe_load(stream)
            #~ data_numpy = numpy.zeros((len(data_loaded), len(data_loaded[0].keys())))
            #~ for index in range(len(data_loaded)):
                #~ data_numpy[index] = data_loaded[index]["X"], data_loaded[index]["Y"], data_loaded[index]["Z"]
            self.points[:] = data_loaded[:]["X"], data_loaded[:]["Y"], data_loaded[:]["Z"]
            
        elif file_extension == ".mat":
            # QUICK HACK
            #~ self.points = scipy.io.loadmat(file_path)["pc"] # inner_sheet.mat
            self.points = scipy.io.loadmat(file_path)["pci"] 
            if self.points.shape[1] > 3:
                raise Exception("Currently only point information can be loaded within this format! Switch to .xyz or .ply format!")
            
        elif file_extension == ".xyz" or file_extension == ".txt":
            self.points = numpy.loadtxt(file_path)
            # We have xyz files with additional normals and maybe curvature as columns
            if self.points.shape[1] > 3 and self.points.shape[1] < 7: 
                self.normals = self.points[:, 3:6]
                self.points = numpy.delete(self.points, [3, 4, 5], axis=1)
            else:
                pass
            if self.points.shape[1] > 6:
                self.normals = self.points[:, 3:6]
                self.curvature = self.points[:, 6]
                self.curvature = self.curvature.reshape((self.curvature.shape[0], 1))
                self.points = numpy.delete(self.points, [3, 4, 5, 6], axis=1)
            else:
                pass

        elif file_extension == ".ply":
            with open(file_path, "rb") as f:   
                plydata = PlyData.read(f)
                
                properties = plydata.elements[0].data.dtype.names
                self.points = numpy.zeros((plydata.elements[0].data.size, 3))
                self.points.T[0], self.points.T[1], self.points.T[2] = plydata.elements[0].data["x"][:], plydata.elements[0].data["y"][:], plydata.elements[0].data["z"][:]
                # We may have more than just point information
                if len(properties) > 3:
                    self.normals = numpy.zeros((plydata.elements[0].data.size, 3))
                    self.normals.T[0], self.normals.T[1], self.normals.T[2] = plydata.elements[0].data["nx"][:], plydata.elements[0].data["ny"][:], plydata.elements[0].data["nz"][:]
                else:
                    pass
                # We may have additional curvature information. Meshlab saves this under "quality"
                if len(properties) > 6:
                    self.curvature = plydata.elements[0].data["quality"]
                    self.curvature = self.curvature.reshape((self.curvature.shape[0], 1))
                else:
                    pass
    
        elif file_extension == ".asc" or ".csv":
            with open(file_path) as f:
                data = csv.reader(f, delimiter=" ")    
                point_list = []
                normals_list = []
                curvature_list = []
                for row in data:
                    point_list.append(row[0:3])
                    if len(row) > 3:
                        normals_list.append(row[3:6])
                    else:
                        pass
                    if len(row) > 6:
                        curvature_list.append(row[6])
                    else:
                        pass
                self.points = numpy.array(point_list, dtype=numpy.float64)
                if normals_list:
                    self.normals = numpy.array(normals_list, dtype=numpy.float64)
                else:
                    pass
                if curvature_list:
                    self.curvature = numpy.array(curvature_list, dtype=numpy.float64)
                else:
                    pass
        
        elif file_extension == ".stl": # This extension might get cancelled and we use meshlab for big data
            model_mesh = mesh.Mesh.from_file(file_path)
            model_mesh.vectors # This will give a triplet of vectors for each vertex
            self.normals = model_mesh.normals
            # TODO Clear this process! Which format do we get from this? We need to delete duplicate points, because we dont care about triangle information!
            ipdb.set_trace() 
            self.points = numpy.vstack((model_mesh.v0, model_mesh.v1, model_mesh.v2))

        else:
            raise Exception("File format not supported. Only input .xyz or .ply point clouds!")
        
        if self.points is None == True:
            raise Exception("Loaded file was empty")

        return
    
    
    def write_pcd(self, file_format, filepath, only_save_points=False):
        """ Write a pcd to a to-be-specified file format """
        if file_format == "ply":
            row_array = self.points
            if only_save_points == True:
                new_array = [tuple(row) for row in row_array.tolist()]
                vertices = numpy.array(new_array, 
                                        dtype=[("x", numpy.float64), ("y", numpy.float64), ("z", numpy.float64)]
                                        )
            else:                
                if self.has_normals() == True:
                    row_array = numpy.hstack((row_array, self.normals))
                else:
                    pass
                if self.has_curvature() == True:
                    row_array = numpy.hstack((row_array, self.curvature))
                else:
                    pass
        
                if row_array.shape[1] == 3:
                    new_array = [tuple(row) for row in row_array.tolist()]
                    vertices = numpy.array(new_array, 
                                            dtype=[("x", numpy.float64), ("y", numpy.float64), ("z", numpy.float64)]
                                            )
                elif row_array.shape[1] == 6:
                    new_array = [tuple(row) for row in row_array.tolist()]
                    vertices = numpy.array(new_array, 
                                            dtype=[("x", numpy.float64), ("y", numpy.float64), ("z", numpy.float64),
                                                   ("nx", numpy.float64), ("ny", numpy.float64), ("nz", numpy.float64)]
                                            )
                elif row_array.shape[1] == 7:
                    new_array = [tuple(row) for row in row_array.tolist()]
                    vertices = numpy.array(new_array, 
                                            dtype=[("x", numpy.float64), ("y", numpy.float64), ("z", numpy.float64),
                                                   ("nx", numpy.float64), ("ny", numpy.float64), ("nz", numpy.float64),
                                                   ("quality", numpy.float64)]
                                            )
                else:
                    raise Exception("Unknown 7th column! Specify values. Currently supported: Points, Normals, Curvature!")
            
            el1 = PlyElement.describe(vertices, "vertex")
            PlyData([el1], text=True).write(filepath)
        
        
        elif file_format == "txt":    
            row_array = self.points
            
            if only_save_points == True:
                numpy.savetxt(filepath, row_array)
            else:                
                if self.has_normals() == True:
                    row_array = numpy.hstack((row_array, self.normals))
                else:
                    pass
                if self.has_curvature() == True:
                    row_array = numpy.hstack((row_array, self.curvature))
                else:
                    pass
                
                numpy.savetxt(filepath, row_array)
            
        
        elif file_format == "yml" or file_format == "yaml":
            data_dict = []
            data_dict.append({"X": float(self.points[:, 0]), "Y": float(self.points[:, 1]), "Z": float(self.points[:, 2])})
            #~ for i in range(self.points.shape[0]):
                #~ data_dict.append({"X": float(self.points[i][0]), "Y": float(self.points[i][1]), "Z": float(self.points[i][2])})
                
            with io.open(filepath, "w", encoding="utf8") as outfile:
                yaml.dump(data_dict, outfile, default_flow_style=False, allow_unicode=True)
        
        else:
            raise TypeError("Specified file format not supported!")
        
        return

    
    def estimate_normals_open3d(self, neighborhood, search_radius):
        """ Estimate normals based on PCA, because we cannot vectorize we rely on the c++ open3d backend """
        pcd_normal = open3d.PointCloud()
        pcd_normal.points = open3d.Vector3dVector(copy.deepcopy(self.points))
        open3d.estimate_normals(pcd_normal, open3d.KDTreeSearchParamHybrid(
                                                        radius = search_radius,
                                                        max_nn = neighborhood)
                                )
        self.normals = numpy.asarray(pcd_normal.normals)
        
        return


    def extract_fpfh(self, radius, max_nn):
        """
        Extract the FastPointFeatureHistograms of a point cloud with Open3D
        
        args:
            radius:     radius for nn-search. Use a much higher radius than for normal computation! e.g. 10xmean_nn
            max_nn:     max number of nearest neighbors within the radius. Use a higher value as well! e.g. 100
        """
        pcd_open3d = open3d.PointCloud()
        pcd_open3d.points = open3d.Vector3dVector(self.points)
        if self.has_normals() == True:
            pcd_open3d.normals = open3d.Vector3dVector(self.normals)
        else:
            raise Exception("Compute normals before computing FPFH!")
        
        fpfh = open3d.compute_fpfh_feature(pcd_open3d, open3d.KDTreeSearchParamHybrid(
                                                            radius=radius, max_nn=max_nn)
                                            )
        self.fpfh = numpy.asarray(fpfh.data).T

        return
    
    
    def find_key_points(self, feature="fpfh", points="variable", quartile=0.6, num_points=100, dist_metric="cityblock"):
        """ 
        Extract points with a high distance to the mu-histogramm. These are more likely to be salient key points 
        
        args:
            feature (str)       :   Feature according to which key points are selected.
            points (str)        :   (Fixed) gives back a fixed number of poins that is given through num_points.  
                                    E.g. 100 gives back 100 key points in total.
                                    (Variable) selects key points outside of the given float set in quartile. 
                                    E.g. 0.9 would return the 0.1 points outside of the confidence interval.
            quartile (float)    :   Float that specifies the interval of rejected points. The higher, the fewer key points will be selected.
            num_points (int)    :   Absolute number of key points that will be selected.
            dist_metric (str)   :   Distance metric for extraction criterion. We use the Manhattan distance as default for FPFH features.
        """
        def compute_distances_to_mu_histogram(self, metric="cityblock"):
            mu_fpfh = numpy.mean(self.fpfh, axis=0).reshape(1, self.fpfh.shape[1])
            if metric == "cityblock":
                dist = scipy.spatial.distance.cdist(self.fpfh, mu_fpfh, metric="cityblock")
            elif metric == "euclidean":
                dist = scipy.spatial.distance.cdist(self.fpfh, mu_fpfh, metric="euclidean")
            elif metric == "kl":            
                dist = numpy.zeros(self.fpfh.shape[0])
                for i in range(self.fpfh.shape[0]):
                    dist[i] = scipy.stats.entropy(self.fpfh[i], mu_fpfh.flatten(), base=None) # Kulback-Leibler divergence
            else:
                raise Exception("Specified metric not in use!")
            
            return dist
        
        if feature == "fpfh":
            fpfh_distances = compute_distances_to_mu_histogram(self, metric=dist_metric)
            dist_mean = numpy.mean(fpfh_distances)
            dist_std = numpy.std(fpfh_distances)
            
            if points == "variable":
                confid_interval = scipy.stats.norm.interval(quartile, loc=dist_mean, scale=dist_std)
                key_points_idx = numpy.where((fpfh_distances < confid_interval[0]) | (fpfh_distances > confid_interval[1]))[0]
            elif points == "fixed":
                if num_points > fpfh_distances.shape[0]:
                    raise Exception("Selected more key points than number of points in PointCloud()")
                else:
                    # This is the fastes partial sort method! The array is sorted so that the n first elements are the largest 
                    # (-a, n[:n])
                    key_points_idx = bottleneck.argpartition(-fpfh_distances.flatten(), num_points)[:num_points]
            else:
                raise Exception("Choose (fixed) or (variable) for points! ")
            
            self.key_points = [self.points[key_points_idx], key_points_idx]
        else:
            raise Exception("Other features are not supported yet!")
        
        return
    
    
    def find_persistent_key_points(self, feature="fpfh", dist_metric="cityblock", scales = [10, 5, 2], neighborhood=100, quartile=0.7):
        """ 
        Persistenc analysis: 
            - Compute features at different radii
            - For each radii, exract key points via their feature distance to the mean feature value
            - Remember key points for each radius
            - Key points that appear in all sets are persistent and extracted
        """ 
        mean_nn = self.get_mean_nn_distance()
        key_sets = []
        for k in scales:
            self.extract_fpfh(radius=k*mean_nn, max_nn=neighborhood)
            self.find_key_points(points="variable", quartile=quartile, dist_metric=dist_metric)
            key_sets.append(self.key_points[1])
        
        persistent_key_points = reduce(numpy.intersect1d, (key_sets[0], key_sets[1], key_sets[2])) # TODO Make this flexible
        self.key_points = [self.points[persistent_key_points], persistent_key_points]
        
        return
    
    
    def uniform_downsample(self, voxel_size):
        pcd_open3d = open3d.PointCloud()
        pcd_open3d.points = open3d.Vector3dVector(copy.deepcopy(self.points))        
        pcd_open3d = open3d.voxel_down_sample(pcd_open3d, voxel_size=voxel_size)
    
        self.points = numpy.asarray(pcd_open3d.points)
    
        return 
    
    
    def filter_outliers(self, neighborhood=20, std_ratio=2.0):
        pcd_open3d = open3d.PointCloud()
        pcd_open3d.points = open3d.Vector3dVector(copy.deepcopy(self.points))
        
        inlier_pcd_open3d, ind = open3d.statistical_outlier_removal(pcd_open3d,
                                             nb_neighbors=neighborhood, std_ratio=std_ratio)
        
        self.points = self.points[ind]
        if self.has_normals() == True:
            self.normals = self.normals[ind]
    
        return
    
    
    def estimate_normals_and_curvature_python(self, neighborhood, search_radius, compute_curvature=False):
        """ 
        Estimate the normals and optionally the curvature of each point in a pointcloud with a defined neighborhood
        
        args:
            neighborhood (int): Maximum number of points to take as neighbors
            search_radius (float): Search radius for the nearest neighbor search
            compute_curvature (bool): optional curvature computation if true
        """
        # ISSUE: This is not vectorizable -> Use a c++ backend solution 
        
        if compute_curvature == True:
            self.curvature = numpy.zeros((self.points.shape[0], 1))
        else:
            pass
            
        self.normals = numpy.zeros((self.points.shape[0], self.points.shape[1]))
        
        for idx, point in enumerate(self.points):
            point = point.reshape((1, -1))
            nbrs = NearestNeighbors(n_neighbors=neighborhood+1, algorithm="kd_tree", metric="euclidean", radius=search_radius).fit(self.points)
            nn_distances, nn_indices = nbrs.radius_neighbors(point, return_distance=True)
            dist, ind = nn_distances[0], nn_indices[0]
            dist, ind = numpy.delete(dist, 0), numpy.delete(ind, 0) # Delete point to get only neighbors
            
            ind = dist.argsort(axis=0) # Sort neighbors by smallest distance
            
            if ind.shape[0] > neighborhood:
                ind = numpy.delete(ind, numpy.arange(neighborhood, ind.shape[0]))
            else:
                pass
            nearest_neighbors = self.points[ind]
            
            covariance_matrix = numpy.cov(nearest_neighbors.T)
            
            eigen_values, eigen_vectors  = numpy.linalg.eig(covariance_matrix)
            min_idx = eigen_values.argmin(axis=0)
            self.normals[idx] = eigen_vectors[min_idx]

            if compute_curvature == True:
                self.curvature[idx] = eigen_values[min_idx] / numpy.sum(eigen_values, axis=0)
            else:
                pass
    

    def transform(self, transformation, **kwargs):
        """
        Transform pcd.points with a given homogenous transformation matrix 
        
        kwargs:
            scale:      addtional scaling
        """
        scale = kwargs.get("scale", 1)
        if scale != 1:
            self.points = scale*self.points
        else:
            pass
                
        R = transformation[:self.points.shape[1], :self.points.shape[1]]
        t = transformation[:-1, -1]
        self.points = self.points @ R.T + numpy.tile(t.T, (self.points.shape[0], 1))
        
        if self.has_key_points() == True:
            self.key_points[0] = self.key_points[0] @ R.T + numpy.tile(t.T, (self.key_points[0].shape[0], 1))
        
        if self.has_normals() == True:
            self.normals = self.normals @ R.T 
        
        
        
        
        return
    
    
    def decenter_pcd(self, old_center):
        """ Decenter the point cloud to the old origin """
        if self.points.shape[1] ==3:
            decenter_trans = numpy.asarray([[1.0, 0.0, 0.0, +old_center[0]],
                                          [0.0, 1.0, 0.0, +old_center[1]],
                                          [0.0, 0.0, 1.0, +old_center[2]],
                                          [0.0, 0.0, 0.0, 1.0]])
        elif self.points.shape[1] ==2:
            decenter_trans = numpy.asarray([[1.0, 0.0, +old_center[0]],
                                          [0.0, 1.0, +old_center[1]],
                                          [0.0, 0.0, 1.0]])
        else:
            raise Exception("Point Cloud dimensions not supported yet!")
        
        self.transform(decenter_trans)
        
        return
        
    def center_around_origin(self, return_mean=False):
        """ Center the point cloud (ndarray) around the coordinate origin """
        mean = numpy.mean(self.points, axis=0)
        if self.points.shape[1] ==3:
            center_trans = numpy.asarray([[1.0, 0.0, 0.0, -mean[0]],
                                          [0.0, 1.0, 0.0, -mean[1]],
                                          [0.0, 0.0, 1.0, -mean[2]],
                                          [0.0, 0.0, 0.0, 1.0]])
        elif self.points.shape[1] ==2:
            center_trans = numpy.asarray([[1.0, 0.0, -mean[0]],
                                          [0.0, 1.0, -mean[1]],
                                          [0.0, 0.0, 1.0]])
        else:
            raise Exception("Point Cloud dimensions not supported yet!")
        
        self.transform(center_trans)
        
        if return_mean == False:
            return
        else:
            return mean
    
    
    def get_bounding_box(self):
        if self.has_points() == False:
            raise Exception("Cannot compute bounding box for empty cloud!")
        else:
            min_coords = numpy.amin(self.points, axis=0)
            max_coords = numpy.amax(self.points, axis=0)
            
            v1 = min_coords[0], min_coords[1], min_coords[2]
            v2 = min_coords[0], min_coords[1], max_coords[2]
            v3 = min_coords[0], max_coords[1], min_coords[2]
            v4 = min_coords[0], max_coords[1], max_coords[2]
            v5 = max_coords[0], min_coords[1], min_coords[2]
            v6 = max_coords[0], min_coords[1], max_coords[2]
            v7 = max_coords[0], max_coords[1], min_coords[2]
            v8 = max_coords[0], max_coords[1], max_coords[2]
            
            self.bounding_box = numpy.array([v1, v2, v3, v4, 
                                             v5, v6, v7, v8]
                                            ) 
            return
    
    def get_mean_nn_distance(self):
        nbrs = NearestNeighbors(n_neighbors=2, algorithm="kd_tree", metric="euclidean").fit(self.points)  
        nn_distances, nn_indices = nbrs.kneighbors(self.points)
    
        return numpy.mean(nn_distances[:, 1])
    
    def has_bounding_box(self):
        if self.bounding_box is not None:
            return True
        else:
            return False
    
    
    def has_points(self):
        if self.points is not None:
            return True
        else:
            return False
    
    
    def has_key_points(self):
        if self.key_points is not None:
            return True
        else:
            return False
    
    
    def has_fpfh(self):
        if self.fpfh is not None:
            return True
        else:
            return False
    
    
    def has_normals(self):
        if self.normals is not None:
            return True
        else:
            return False
    
    
    def has_curvature(self):
        if self.curvature is not None:
            return True
        else:
            return False
    
    
    def has_color(self):
        if self.color is not None:
            return True
        else:
            return False
