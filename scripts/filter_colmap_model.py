#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 23:40:52 2020

@author: vik748
"""
import pandas as pd
import os
import json
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import sys,os

def _make_gen(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024*1024)

def rawgencount(filename):
    f = open(filename, 'rb')
    f_gen = _make_gen(f.raw.read)
    return sum( buf.count(b'\n') for buf in f_gen )

def plot_sphere(axes, center, radius):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)*radius + center[0]
    y = np.sin(u)*np.sin(v)*radius + center[1]
    z = np.cos(v)*radius + center[2]
    axes.plot_wireframe(x, y, z, color="r")
    
def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def set_axes_equal(ax, limits=None):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    if limits is None:
        limits = np.array([ ax.get_xlim3d(),
                            ax.get_ylim3d(),
                            ax.get_zlim3d()  ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)

def find_pointcloud_std_inliers(coords, std_factor = 3, plot=False):
    '''
    Given an Nx3 numpy array with point cloud coordinates, return boolean array of inliers
    
    '''
    
    inliers = np.ones((len(coords)),dtype=bool)
    center = np.array([0,0,0])
    dists_mean = 1000
    no_inliers = len(coords)
    change_in_center = 1e9
    
    while change_in_center > 1e-9 :
        last_center = center
        last_dists_mean = dists_mean
        last_no_inliers = no_inliers
            
        center = np.mean(coords[inliers],axis=0)
        print("Center: ",center)
        
        dists = np.linalg.norm(coords-center,axis=1)
        dists_mean = dists[inliers].mean()
        dists_std = dists[inliers].std()
        
        inliers = np.logical_and(inliers, dists <= ( dists_mean + std_factor * dists_std ) )
        no_inliers = np.sum(inliers)
    
        change_in_center = np.linalg.norm(last_center - center)
        change_in_mean = last_dists_mean - dists_mean
    
        print("Inliers: {} Mean dist: {:.2f} Std dev: {:.2f} Center delta: {:.2f} Mean delta:{:.2f}".format(
                            no_inliers, dists_mean, dists_std, change_in_center, change_in_mean))
    
    if plot:               
        fig = plt.figure(1)
        ax = plt.axes(projection="3d")
        
        ax.scatter3D(center[0],center[1],center[2],s=5)
        ax.scatter3D(coords[inliers,0],coords[inliers,1],coords[inliers,2],s=1)
        plot_sphere(ax,center,dists_mean + std_factor * dists_std)
        set_axes_equal(ax)
        #ax.set_aspect('equal')
        
        plt.show()
    return inliers

colmap_model_fold = "/data/colmap_data/Morgan2_073019/models_4_txt_5/"

images_file = os.path.join(colmap_model_fold, 'images.txt')
points_file = os.path.join(colmap_model_fold, 'points3D.txt')

nlines = rawgencount(images_file)

skip_row_list = [0,1,2,3] + list(range(5,nlines,2))
cams_df = pd.read_csv(images_file, sep=' ', skiprows=skip_row_list, index_col='IMAGE_ID',
                      names= ['IMAGE_ID', 'QW', 'QX', 'QY', 'QZ', 'TX', 'TY', 'TZ', 'CAMERA_ID', 'NAME'])

cams_df = pd.read_csv(images_file, sep=' ', skiprows=skip_row_list, index_col='IMAGE_ID',
                      names= ['IMAGE_ID', 'QW', 'QX', 'QY', 'QZ', 'TX', 'TY', 'TZ', 'CAMERA_ID', 'NAME'])

points_df = pd.read_csv(points_file, sep=' ', skiprows=3, header=None, usecols=[0,1,2,3,4,5,6,7], index_col='POINT3D_ID',
                        names= ['POINT3D_ID', 'X', 'Y', 'Z', 'R', 'G', 'B', 'ERROR'])

coords = points_df[['X','Y','Z']].values
inliers = find_pointcloud_std_inliers(coords, plot=True)
nlines = rawgencount(points_file)
header_lines = 3
inliers_file = np.insert(inliers, 0, np.ones(header_lines))

assert len(inliers_file) == nlines
with open(points_file) as f:
    data = f.readlines()
  
points_file_out = os.path.join(colmap_model_fold, 'points3D_out.txt')

with open(points_file_out, 'w') as f:
    for l, inlier in zip(data,inliers_file):
        if inlier:
            f.write(l)
            
with open(images_file) as f:
    data = f.readlines()

header_lines = 4

images_file_out = os.path.join(colmap_model_fold, 'images_out.txt')
pt3d_outlier_ids = set(points_df.index[~inliers])

with open(images_file_out, 'w') as fout:
    with open(images_file) as fin: 
    
        for i in range(header_lines):
            line = next(fin)
            fout.write(line)
               
        print(line)
        for line in fin:
            #print(line)
            fout.write(line)
            
            # pts line
            line = next(fin)
            l_split = line.strip().split(' ')
            
            assert len(l_split) % 3 == 0
            line_out = ''
            for x,y,pt3d_id in zip(l_split[0::3], l_split[1::3], l_split[2::3]):
                if pt3d_id != '-1' and int(pt3d_id) in pt3d_outlier_ids:
                    line_out += ' '.join([x, y, '-1']) + ' '
                    #print("removed: ", pt3d_id)
                else:
                    line_out += ' '.join([x, y, pt3d_id]) + ' '
            fout.write(line_out.strip(' ') + '\n')
                
            