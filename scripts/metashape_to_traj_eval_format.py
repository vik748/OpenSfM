#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 16:36:22 2020

@author: vik748
"""
import numpy as np
from transforms3d.quaternions import mat2quat, axangle2quat
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import json
import cv2
import os 
np.set_printoptions(suppress=True)

def compose_T(R,t):
    return np.vstack((np.hstack((R,t)),np.array([0, 0, 0, 1])))

def decompose_T(T_in):
    return T_in[:3,:3], T_in[:3,[-1]].T

def pose_inv(R_in, t_in):
    t_out = -np.matmul((R_in).T,t_in)
    R_out = R_in.T
    return R_out,t_out

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

filename = "/home/vik748/Downloads/Stingray2_08072018_every_10_camera_poses_in_mts_20200910.txt"
name, ext = os.path.splitext(filename)
out_filename = name+'_out'+ext

inames, home_pose_array = read_metashape_poses(filename)
with open(out_filename,'w') as out_file:
    out_file.write('# time x y z qx qy qz qw\n')
    for iname, hom_pose in zip(inames, home_pose_array):
        R = hom_pose[:3,:3]
        t = hom_pose[:3,-1]
        qw, qx, qy, qz = mat2quat(R)
        output_array = np.append(t, [qx, qy, qz, qw])
        output_string = np.array2string(output_array,separator=' ',max_line_width=1000,precision=16)[1:-1]
        i_time = float(re.sub(r"\D", "", iname))
        output_line = '{:.3f} '.format(i_time)+output_string+'\n'
        output_line = re.sub(' +', ' ', output_line)
        output_line = output_line.replace(' \n','\n')
        out_file.write(output_line)
        
recon_file = '/home/vik748/Downloads/reconstructionStingray2_HAHOG.json'
name, ext = os.path.splitext(recon_file)
out_filename = name+"_out.txt"

with open(recon_file) as f:
    data = json.load(f)
    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')         # important!
title = ax.set_title('3D Test')
    
shots = data[0]['shots']
camera_names = sorted(shots.keys())
with open(out_filename,'w') as out_file:
    out_file.write('# time x y z qx qy qz qw\n')
    for cam_name, shot in sorted(shots.items()):
        Raxis_world = np.array(shot['rotation'])
        t_world = np.array(shot['translation'])
        #R = axangle2mat(Raxis, np.linalg.norm(Raxis))
        R_world,_ = cv2.Rodrigues(Raxis_world)
        H = compose_T(*pose_inv(R_world, t_world[:,np.newaxis])) # camera pose in hom coordinates
        
        R, t = decompose_T(H)
        
        qw, qx, qy, qz = mat2quat(R)
        output_array = np.append(t, [qx, qy, qz, qw])
        output_string = np.array2string(output_array,separator=' ',max_line_width=1000,precision=16)[1:-1]        
        i_time = float(re.sub(r"\D", "", cam_name))
        output_line = '{:.3f} '.format(i_time)+output_string+'\n'
        output_line = re.sub(' +', ' ', output_line)
        output_line = output_line.replace(' \n','\n')
        out_file.write(output_line)
        
        plot_pose3_on_axes(ax, H, axis_length=.5)
        set_axes_equal(ax)
        plt.pause(.1)