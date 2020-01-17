#!/usr/bin/env python3

"""
Super4PCs registration
"""

import os
import numpy 
import subprocess
import re
import ipdb
import copy
import time
from pyreg.local_reg.local_registration_base import ConvergenceCriteria
import pyreg.functions
from pyreg.point_cloud_class import *
from pyreg.registration_base import RegistrationBase


# TODO Throw exceptions for conflicting args
class Super4PCSRegistration(RegistrationBase):
    def __init__(self, scan, model, scan_filepath, model_filepath, output_path=None, ConvergenceCriteria=None, 
                overlap=0.8, normal_diff=None, algorithm="Super4PCS", num_samples=None, max_time=1000, 
                write_temp_files=False, write_result=False):    
        RegistrationBase.__init__(self, scan, model)
        
        self.output_path = None if output_path is None else output_path
        self.overlap = overlap
        self.algorithm = algorithm
        self.normal_diff = normal_diff
        self.samples = num_samples
        self.max_time = max_time
        self.tolerance = None if ConvergenceCriteria is None else ConvergenceCriteria.tolerance
        self.num_inliers = 0.0
        self.time = 0.0
        self.transformation = numpy.eye(4)
        self.write_result = write_result
        
        if write_temp_files == False:        
            self.target_path = os.path.normpath(model_filepath) 
            self.source_path = os.path.normpath(scan_filepath)
        else:
            self.write_temp_files()
        
        
    def write_temp_files(self):
        """ Write new temp files from input with the number of points in the first line """
        # Write temp files of new scaled pcd for registration
        source_temp_path = self.output_path+"/source_temp.ply"
        target_temp_path = self.output_path+"/target_temp.ply"
        # Change paths to the temp ones, so that we register the modified PointCloud
        self.source_path = source_temp_path
        self.target_path = target_temp_path
        
        # TODO throw exception when PointCloud has curvature!
        self.SCAN.write_pcd(file_format="ply", filepath=source_temp_path, only_save_points=True)
        self.TARGET.write_pcd(file_format="ply", filepath=target_temp_path, only_save_points=True)
        
        return
    
    
    def register(self):
        """
        Register scan onto model by using the precompiled super4PCs.exe
        
        args:
            -o      overlap between the point clouds  
            -d      Epsilon band for rejection
            -n      number of samples used for pose generation (For good scans and high overlap, we can use only few samples like 500)
            -a      normal orientation difference: Select something like 90.0 
            -x      False by default, when naming it is True
        """
        exe_filepath = "~/Documents/Masterarbeit/Code/virt_env/env/lib/python3.6/site-packages/OpenGR/build/install/bin/"
        
        # TODO raise Exception when file format is not .ply or .obj
        #~ cmd = "./Super4PCS -i "+self.target_path+" "+self.source_path+" -o "+str(self.overlap)# + " -t "+str(self.max_time)
        cmd = "./Super4PCS -i "+self.source_path+" "+self.target_path+" -o "+str(self.overlap)# + " -t "+str(self.max_time)
        if self.samples is not None:
            cmd = cmd + " -n "+ str(self.samples)
        else:
            pass
        
        if self.tolerance is not None:
            cmd = cmd + " -d "+ str(self.tolerance)
        else:
            pass
        
        if self.normal_diff is not None:
            cmd = cmd + " -a "+ str(self.normal_diff)
        else:
            pass
            
        if self.write_result == True:
            cmd = cmd + " -r "+ str(self.output_path+"registered_source.obj")
        else:
            pass    
        
        # We can also use 4PCS
        if self.algorithm == "4PCS":
            cmd = cmd + " -x true"
        else:
            pass
        
        ipdb.set_trace()
        
        abs_cmd = exe_filepath+cmd
        
        start = time.time()
        result = subprocess.check_output(abs_cmd, shell=True).decode('utf-8')
        end = time.time()
        self.time = end-start
        
        ipdb.set_trace()
        
        last_part = result[result.rfind("Score: "):]
        last_error_line = last_part.splitlines()[0]
        floats_in_str = [float(s) for s in re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", last_error_line)]
        self.num_inliers = floats_in_str[0]
        
        result = result.split(self.target_path, 1)[1] # The script reports every iteration! We split once the transformation result is given after model filepath
        result = result.split("Exporting", 1)[0] 
        
        transformation_data = []
        for element in result.split():
            try:
                transformation_data.append(float(element))
            except ValueError:
                pass
        
        # Turn str sequence into numpy array
        for idx, element in enumerate(transformation_data):
            transformation_data[idx] = float(element)
        
        rotation = numpy.eye(3)
        rotation[0, :], rotation[1, :], rotation[2, :] = transformation_data[0:3], transformation_data[4:7], transformation_data[8:11] 
        translation = numpy.zeros((3, 1))
        translation[0], translation[1], translation[2] = transformation_data[3], transformation_data[7], transformation_data[11]
        
        transformation = numpy.hstack((rotation, translation))
        transformation = numpy.vstack((transformation, numpy.array([[0.0, 0.0, 0.0, 1.0]])))
    
        self.transformation = transformation
        
        return

