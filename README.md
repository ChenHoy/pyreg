# PyReg: A Point Cloud Registration Package for Python
author: Christian Homeyer

# What is this package
The pyreg registration package allows efficient registration of even larger 3D point clouds. It includes self written local registration algorithms (ICP, CPD) written entirely in Python and a collection of global registration algorithm from several backends in C++.  
The background of the project was my master thesis.

# Dependencies
This package is not self contained and requires building the separate packages for:  
- Mayavi (visualization) https://github.com/enthought/mayavi 
- Super4PCS https://github.com/STORM-IRIT/OpenGR
- Go-ICP  https://github.com/yangjiaolong/Go-ICP
- Open3D (visualization and essential functions) https://github.com/intel-isl/Open3D 
- Numerous python packages (Found in future requirements.txt for pip)

# What can be done
The package allows several core functions:  
- Input/Output methods for various PCL formats  
- Rigid manipulation of point clouds 
- Feature computations
- Visualization of several properties
- Local registration algorithms with a number of tuning parameters
	- ICP variants with self written solvers
	- Coherent Point Drift (CPD) as a GMM
- Global registration algorithms
	- RANSAC with invariant features
	- Super4PCS  
	- GO-ICP  
	- Fast Graduated Convexity approach
- Benchmark scripts
- Downsampled datasets:
	- Stanford bunny  
	- Stanford dragon
# TODO
1. Code has not been revised and project may not be buildable!!! See how to build the other packages on the github pages before importing the python module.  
2. Make requirements.txt file for pip. 
3. Module is also not shippable as of now. Preparing a cmake build or a wheel would be nice
4. Documentation could be improved.
5. Some dependencies could be resolved via wrappers. There exists a cython wrapper for Go-ICP now.

