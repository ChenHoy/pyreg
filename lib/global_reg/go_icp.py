#!/usr/bin/env python3

"""
Go-ICP registration
"""

import os
import numpy 
import subprocess
import re
import ipdb
import copy
import time
import pyreg.functions
import scipy.spatial.distance
import scipy.stats
from sklearn import preprocessing
from pyreg.point_cloud_class import *
from pyreg.registration_base import RegistrationBase


class GoICPRegistration(RegistrationBase):
    """
    Register source onto target by using the precompiled ./GoICP. The convergence threshold and trim ratio of the trimmed ICP have to 
    be set in the config_go_icp.txt file from config_path! 
    
    args:
        - source        :   point cloud of the scan
        - target        :   point cloud of the model
        -target_path    :   path to temp target data
        -source_path    :   path to temp source data
        -num_points     :   sample of points. Go-ICP uses the first N points of the file in the .exe. 
                            We therefor random sample to create no bias.
        -config_path    :   path to config file 
        -output_path    :   path to output file     
    """
    def __init__(self, source, target, target_filepath, source_filepath, config_path, output_path, epsilon, num_samples=None, overlap=1.0):
        def change_config(config_path, overlap, epsilon):
            """ Change overlap and epsilon in the config file if they are given as an option """
            def replace_line(file_path, line_num, new_text):
                with open(file_path, 'r') as data:
                    lines = data.readlines()
                    lines[line_num] = new_text
                
                with open(file_path, 'w') as out:
                    out.writelines(lines)
                
                return

            # Change overlap in line 25: in the config file we give the trimming percentage (1-overlap)
            replace_line(self.config_path, 24, "trimFraction="+str(1-self.overlap)+"\n") 
            replace_line(self.config_path, 3, "MSEThresh="+str(self.epsilon)+"\n")
            
            return
    
        
        def write_temp_files(source, target, temp_path, downsample=False):
            """ Write new temp files from input with the number of points in the first line """
            def add_num_points_to_first_line(filepath, num_points):
                """ Our temp files need the number of points in the first line as an integer """
                with open(filepath, 'r') as original: 
                    data = original.read()
                
                with open(filepath, 'w') as modified: 
                    new_first_line = "%d \n" %(num_points)
                    modified.write(new_first_line+data)
                
                return        

            # Write temp files of new scaled pcd for registration
            source_temp_path = self.temp_path+"/source_temp.xyz"
            target_temp_path = self.temp_path+"/target_temp.xyz"
            
            if downsample == False:                
                source.write_pcd(file_format="txt", filepath=source_temp_path, only_save_points=True)
                target.write_pcd(file_format="txt", filepath=target_temp_path, only_save_points=True)
            else:
                # Random shuffle the points, because Go-ICP just takes the first n points
                new_SCAN_idx = numpy.arange(self.N)
                numpy.random.shuffle(new_SCAN_idx)
                new_TARGET_idx = numpy.arange(self.M)
                numpy.random.shuffle(new_TARGET_idx)
                
                source.points = source.points[new_SCAN_idx]
                target.points = target.points[new_TARGET_idx]
                
                source.write_pcd(file_format="txt", filepath=source_temp_path, only_save_points=True)
                target.write_pcd(file_format="txt", filepath=target_temp_path, only_save_points=True)
            
            # This is requested by the Go-ICP algorithm
            add_num_points_to_first_line(source_temp_path, self.N)
            add_num_points_to_first_line(target_temp_path, self.M)
            
            return
        
        
        def scale_to_unit_box(source, target):
            """ 
            Scale data, so that each coordinate fits into [-1,1]³
            """    
            mean_source = source.center_around_origin(return_mean=True)
            mean_target = target.center_around_origin(return_mean=True)
            
            source.get_bounding_box()
            target.get_bounding_box()
            
            all_values = numpy.array([source.bounding_box, target.bounding_box]).flatten()
            max_idx = numpy.argmax(abs(all_values))
            scale = abs(all_values[max_idx])
            
            # Scale points with bigger range of the two point clouds
            source.points = source.points/scale
            target.points = target.points/scale
            
            # Update bounding box
            source.get_bounding_box()
            target.get_bounding_box()
            
            return scale, mean_source, mean_target
        
        RegistrationBase.__init__(self, source, target)
        self.num_samples = self.N if num_samples is None else num_samples
        
        self.source_path = os.path.normpath(source_filepath)
        self.target_path = os.path.normpath(target_filepath)
        self.temp_path = os.path.split(source_filepath)[0]
        self.output_path = os.path.normpath(output_path)
        self.config_path = os.path.normpath(config_path)
        
        self.scale, self.mean_source, self.mean_target = scale_to_unit_box(self.SCAN, self.TARGET)
        # When this is smaller, then we downsample
        if self.num_samples < self.N:
            write_temp_files(self.SCAN, self.TARGET, self.temp_path, downsample=True)
        else:
            write_temp_files(self.SCAN, self.TARGET, self.temp_path)
        
        self.overlap = overlap
        self.epsilon = epsilon
        change_config(self.config_path, self.overlap, self.epsilon)    
        self.num_icp = 0
        self.time = 0.0             # This is the actual time we measure for running the .exe (go-icp +     
        self.go_icp_time = 0.0      # This is the time from the command line given by GO-ICP (without Preprocessing)
        self.fitness = 10.0         # This is just so it's not zero
        self.transformation = numpy.eye(4)
    
    
    def register(self):
        def read_output_file(filepath):
            with open(filepath, 'r') as original:  
                data = original.readlines()
                data = [x.strip() for x in data]
                
            # First line is the total_time of the registration
            time = float(data[0])
            
            # Rest is transformation data
            transformation_data = data[1:] 
            rotation_data = transformation_data[:3]
            translation_data = transformation_data[3:]
            for idx, element in enumerate(translation_data):
                translation_data[idx] = float(element)
            translation = numpy.array(translation_data)
            
            rotation = numpy.eye(3)
            for idx, element in enumerate(rotation_data):
                row = []
                for value in element.split():
                    try:
                        row.append(float(value))
                    except ValueError:
                        pass
                        
                rotation[idx, :] = row
            
            transformation = numpy.hstack((rotation, numpy.reshape(translation, (3, 1))))
            transformation = numpy.vstack((transformation, numpy.array([[0.0, 0.0, 0.0, 1.0]])))
            
            return time, transformation
        
        exe_filepath = "~/Documents/Masterarbeit/Code/virt_env/env/lib/python3.6/site-packages/Go-ICP/build/"
        source_temp_path = self.temp_path+"/source_temp.xyz"
        target_temp_path = self.temp_path+"/target_temp.xyz"
        
        rel_cmd = "./GoICP "+target_temp_path+" "+source_temp_path+" "+str(self.num_samples)+" "+self.config_path+" "+self.output_path
        abs_cmd = exe_filepath+rel_cmd    
        
        start = time.time()
        cmd_lines = subprocess.check_output(abs_cmd, shell=True).decode('utf-8')
        end = time.time()
        self.time = end-start
        
        self.num_icp = cmd_lines.count("ICP ", 0, len(cmd_lines))
        last_part = cmd_lines[cmd_lines.rfind("Error*: "):]
        last_error_line = last_part.splitlines()[0]
        floats_in_str = [float(s) for s in re.findall(r'\b\d+\b', last_error_line)]
        self.fitness = floats_in_str[0]
        
        self.go_icp_time, self.transformation = read_output_file(self.output_path) 
        
        # Adjust the transformation for the unscaled and uncentered data 
        t = self.transformation[:-1, -1] 
        R = self.transformation[:self.D, :self.D]
        self.transformation[:-1, -1] = self.scale*t + self.mean_target.T - R @ self.mean_source.T
        
        try:  # Delete temp files
            os.system("rm "+source_temp_path)
            os.system("rm "+target_temp_path)
        finally:
            pass
        
        return 


