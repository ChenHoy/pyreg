import numpy   
from random import  uniform
import matplotlib.pyplot
import scipy.spatial


"""
This tool file contains general methods for point set registration
"""


def convexhull(point_set):
  """
  Compute the convex hull of a point set

  args
    point_set (ndarray): Input point set
  return
    point_set (ndarray): < p with convex hull points
  """
  point_set = numpy.array(point_set)
  hull = scipy.spatial.ConvexHull(point_set)
  
  return point_set[hull.vertices, :]


def ccw_sort(point_set):
  """
  Compute the concave hull of a point set.

  args
    point_set (ndarray): input point set
  return
    point_set (ndarray): sorted output point set for plotting
  """
  point_set = numpy.array(point_set)
  mean = numpy.mean(point_set, axis=0)
  shifted_set = point_set - mean
  new_set = numpy.arctan2(shifted_set[:, 0], shifted_set[:, 1])
  
  return point_set[numpy.argsort(new_set), :]


def gen_ran_poly(num_points, dim):
  """ Generate a convex polygone from n-dim point sets """
  ran_points = numpy.random.rand(num_points, dim)
  polygon = convexhull(ran_points)
  
  return polygon


def center_point_set(point_set):
  """ Center a point set around origin """
  mean_coord = numpy.mean(point_set, 0)
  x_mean = mean_coord[0]
  y_mean = mean_coord[1]
  point_set[:, 0] = point_set[:, 0] - x_mean
  point_set[:, 1] = point_set[:, 1] - y_mean
  
  return point_set


def rotation_two(angle):
  """ Generate rotation matrix in 2D with numpy """
  cosine, sine = numpy.cos(angle), numpy.sin(angle)
  rotation_matr = numpy.array(((cosine, -sine), (sine, cosine)))
  
  return rotation_matr


def gen_ran_rot(angle_min, angle_max):
  """ Generate a random rotation """
  angle = uniform(angle_min, angle_max)
  ran_rotation, angle = rotation_two(angle)
  
  return ran_rotation, angle


def plot_poly(point_set, title, xlimit, ylimit):
  """ Plot a polygon of a point set with specified options """
  fig, axes = matplotlib.pyplot.subplots()
  axes.set_title(title)
  axes.set_xlim(xlimit)
  axes.set_ylim(ylimit)
  poly = matplotlib.pyplot.Polygon(point_set, ec="k")
  axes.add_patch(poly)
  x_coord, y_coord = zip(*point_set)
  axes.scatter(x_coord, y_coord, color="k", alpha=0.6, zorder=3)
  
  return
