#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 21:22:45 2020

@author: auv

ex image_dataset_structure:
    └── Stingray2_080718
        ├── Stingray2_080718_3_bit
        ├── Stingray2_080718_4_bit
        ├── Stingray2_080718_5_bit
        ├── Stingray2_080718_6_bit
        ├── Stingray2_080718_7_bit
        └── Stingray2_080718_RAW

starting processing structure:
    └── opensfm_processing
        └── Stingray2_080718
            ├── camera_models_overrides.json
            └── config_template.yaml

resulting output structure
 
"""
import sys,os
import logging
import numpy as np
import re
from shutil import copy, copytree, move
import yaml
#import subprocess #import PIPE, run, STDOUT        
from subprocess import Popen, PIPE, CalledProcessError
from opensfm import log
import json
import pandas as pd
import time
from transforms3d.quaternions import mat2quat
from transforms3d.axangles import axangle2mat
from datetime import datetime

def execute_cmd(command):    
    with Popen(command, stdout=PIPE, stderr=PIPE, bufsize=1, universal_newlines=True) as p:
        output = ""
        for line in p.stderr:
            #print("\r\t>>> "+line, end='') # process line here
            output+=line
            
    if p.returncode != 0:
        print("\n\t>>> ".join(output.split('\n')[-20:]))
        raise CalledProcessError(p.returncode, p.args)
    return p.returncode, output

def recon_results_summary(recon_file):
    with open(recon_file) as f:
        data = json.load(f)
    
    results = ""
    image_names = ""
    results = results + "Number of models generated: {:d}".format(len(data))
    i=0
    res_dict = {"num_models": len(data)}
    
    for i,model in enumerate(data):
        results += "\n\tModel {:d}:-".format(i)
        results += "\tImages: {:d}".format(len(model['shots']))
        results += "\tPoints: {:d}".format(len(model['points']))
        image_names += "\n\tModel {:d}:-".format(i)
        for image_name in model['shots'].keys():
            image_names += " " + image_name
        res_dict.update({"model_{:d}_num_cams".format(i) : len(model['shots']),
                         "model_{:d}_num_points".format(i) : len(model['points']) })        
        
    return results, image_names, res_dict

def evo_ape_results(stats_json):
    with open(stats_json) as f:
        data = json.load(f)  
        
    return data

def tracks_results_summary(recon_file):
    with open(recon_file) as f:
        data = json.load(f)
        
    return data['num_tracks']

def compose_T(R,t):
    return np.vstack((np.hstack((R,t)),np.array([0, 0, 0, 1])))

def decompose_T(T_in):
    return T_in[:3,:3], T_in[:3,[-1]].T

def pose_inv(R_in, t_in):
    t_out = -np.matmul((R_in).T,t_in)
    R_out = R_in.T
    return R_out,t_out

def world_axangle_t_2_object_pose(axis_angle_world, transl_world):
    '''
    Convert pose representated as axis angle (axis_angle) and translation(transl) of world origin
    to homogenous matrix of object in world origin
    This is as it applies to the opencv PNP algorithm
    '''
    #R_world,_ = cv2.Rodrigues(axis_angle_world)
    #the following is equivalent to opencv function above
    R_world = axangle2mat(axis_angle_world, np.linalg.norm(axis_angle_world))
    R_obj, transl_obj = pose_inv(R_world, transl_world[:,np.newaxis])
    H_world_obj = compose_T(R_obj, transl_obj) # camera pose in hom coordinates
    return H_world_obj

def world_axangle_t_2_object_poseRT(axis_angle_world, transl_world):
    '''
    Convert pose representated as axis angle (axis_angle) and translation(transl) of world origin
    to homogenous matrix of object in world origin
    This is as it applies to the opencv PNP algorithm
    '''
    #R_world,_ = cv2.Rodrigues(axis_angle_world)
    #the following is equivalent to opencv function above
    R_world = axangle2mat(axis_angle_world, np.linalg.norm(axis_angle_world))
    R_obj, transl_obj = pose_inv(R_world, transl_world[:,np.newaxis])    
    return R_obj, transl_obj

def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)

def plot_pose3_on_axes(axes, T, axis_length=0.1, center_plot=False, line_obj_list=None, zoom_to_fit=False):
    """Plot a 3D pose 4x4 homogenous transform  on given axis 'axes' with given 'axis_length'."""
    return plot_pose3RT_on_axes(axes, *decompose_T(T), axis_length, center_plot, line_obj_list, zoom_to_fit=zoom_to_fit)

def plot_pose3RT_on_axes(axes, gRp, origin, axis_length=0.1, center_plot=False, line_obj_list=None, zoom_to_fit=False):
    """Plot a 3D pose on given axis 'axes' with given 'axis_length'."""
    # draw the camera axes
    x_axis = origin + gRp[:, 0] * axis_length
    linex = np.append(origin, x_axis, axis=0)

    y_axis = origin + gRp[:, 1] * axis_length
    liney = np.append(origin, y_axis, axis=0)

    z_axis = origin + gRp[:, 2] * axis_length
    linez = np.append(origin, z_axis, axis=0)


    if line_obj_list is None:
        xaplt = axes.plot(linex[:, 0], linex[:, 1], linex[:, 2], 'r-')
        yaplt = axes.plot(liney[:, 0], liney[:, 1], liney[:, 2], 'g-')
        zaplt = axes.plot(linez[:, 0], linez[:, 1], linez[:, 2], 'b-')

        if center_plot:
            center_3d_plot_around_pt(axes,origin[0],zoom_to_fit=zoom_to_fit)
        return [xaplt, yaplt, zaplt]

    else:
        line_obj_list[0][0].set_data(linex[:, 0], linex[:, 1])
        line_obj_list[0][0].set_3d_properties(linex[:,2])

        line_obj_list[1][0].set_data(liney[:, 0], liney[:, 1])
        line_obj_list[1][0].set_3d_properties(liney[:,2])

        line_obj_list[2][0].set_data(linez[:, 0], linez[:, 1])
        line_obj_list[2][0].set_3d_properties(linez[:,2])

        if center_plot:
            center_3d_plot_around_pt(axes,origin[0],zoom_to_fit=zoom_to_fit)
        return line_obj_list

def read_metashape_poses(file):
    '''
    This function take a file with camera poses with name, 4x4 T matrix as a 1x16 row
    and returns a list of img_names and N x 4 x 4 numpy array of Poses
    '''
    img_names = []
    #pose_array = np.zeros([0,4,4])
    with open(file) as f:
        first_line = f.readline()
        if not first_line.startswith('Image_name,4x4 Tmatrix as 1x16 row'):
            raise ValueError("File doesn't start with 'Image_name,4x4 Tmatrix as 1x16 row' might be wrong format")
        data = f.readlines()
        pose_array = np.zeros([len(data),4,4])
        for i,line in enumerate(data):
            name, T_string = (line.strip().split(',',maxsplit=1))
            T = np.fromstring(T_string,sep=',').reshape((4,4))
            img_names.append(name)
            pose_array[i] = T
    return img_names, pose_array

def validate_date(date_text, date_format='%Y-%m-%d-%H-%M-%S'):
    try:
        datetime.strptime(date_text, date_format)
        return True
    except ValueError:
        return False

def camname2time(name):
    if name.startswith('ESC'):
        # For skerki dataset
        return float(name.split('.')[-1])
    elif name.startswith('G'):
        # For gopro datasets
        return float(name.replace('G',''))
    elif validate_date(name.split('_')[0],'%Y-%m-%d-%H-%M-%S'):
        # For UAV IR datasets
        return float(name.split('_')[1].split('-')[1])
    else:
        return None
        
        
def metashapeposes2tum(ms_file, tum_file=None):
    '''
    This function converts metashape pose export file which is 'Image_name,4x4 Tmatrix as 1x16 row'
    to tum format for evo and rpg_trajector_evaluation packages as 'time x y z qx qy qz qw'
    '''    
    if tum_file is None:
        #filename = "/home/vik748/Downloads/Stingray2_08072018_every_10_camera_poses_in_mts_20200910.txt"
        name, ext = os.path.splitext(ms_file)
        tum_file = name+'_tum'+ext

    inames, home_pose_array = read_metashape_poses(ms_file)
    with open(tum_file, 'w') as out_file:
        out_file.write('#time x y z qx qy qz qw\n')
        for iname, hom_pose in zip(inames, home_pose_array):
            R = hom_pose[:3,:3]
            t = hom_pose[:3,-1]
            qw, qx, qy, qz = mat2quat(R)
            output_array = np.append(t, [qx, qy, qz, qw])
            output_string = np.array2string(output_array,separator=' ',max_line_width=1000,precision=16)[1:-1]
            #print(iname)
            i_time = camname2time(iname)
            output_line = '{:.3f} '.format(i_time)+output_string+'\n'
            output_line = re.sub(' +', ' ', output_line)
            output_line = output_line.replace(' \n','\n')
            out_file.write(output_line)
    return 0

def opensfmposes2tum(opensfm_json, tum_file=None, plot_poses = False):
    '''
    This function converts opensfm reconstruction json file
    to tum format for evo and rpg_trajector_evaluation packages as 'time x y z qx qy qz qw'
    '''    
    if tum_file is None:
        name, ext = os.path.splitext(opensfm_json)
        tum_file = name+'_tum.txt'

    #recon_file = '/home/vik748/Downloads/reconstructionStingray2_HAHOG.json'

    with open(opensfm_json) as f:
        data = json.load(f)
        
    if plot_poses:        
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect('equal')         # important!
        title = ax.set_title('3D Test')
        
        
    with open(tum_file, 'w') as out_file:
        out_file.write('#time x y z qx qy qz qw\n')

        if len(data) > 0:
            shots = data[0]['shots']
            #camera_names = sorted(shots.keys())
                       
            for cam_name, shot in sorted(shots.items()):
                axis_angle_world = np.array(shot['rotation'])
                t_world = np.array(shot['translation'])
                
                R, t = world_axangle_t_2_object_poseRT(axis_angle_world, t_world)
                
                qw, qx, qy, qz = mat2quat(R)
                output_array = np.append(t, [qx, qy, qz, qw])
                output_string = np.array2string(output_array,separator=' ',max_line_width=1000,precision=16)[1:-1]                    
                i_time = camname2time(os.path.splitext(cam_name)[0])
                #print (cam_name, i_time)
                output_line = '{:.3f} '.format(i_time)+output_string+'\n'
                output_line = re.sub(' +', ' ', output_line)
                output_line = output_line.replace(' \n','\n')            
                out_file.write(output_line)
                
                if plot_poses:
                    plot_pose3RT_on_axes(ax, R, t.T, axis_length=.5)
                    set_axes_equal(ax)
                    plt.pause(.1)    
            return 0
        else:
            return 1