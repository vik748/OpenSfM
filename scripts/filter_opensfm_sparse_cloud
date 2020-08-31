#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 12:15:33 2020

@author: vik748
"""
import json
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import sys,os

# draw sphere
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

recon_file = '/home/vik748/Downloads/reconstruction.json'
name, ext = os.path.splitext(recon_file)
output_recon_file = name+"_filtered"+ext


with open(recon_file) as f:
  data = json.load(f)

points_dict = data[0]['points']

coords = np.zeros((len(points_dict),3))

for i,track_id in enumerate(sorted(points_dict.keys(),key = int)):
    coords[i,:] = np.array(points_dict[track_id]['coordinates'])

inliers = np.ones((len(coords)),dtype=bool)
center = np.array([0,0,0])
dists_mean = 1000
no_inliers = len(coords)
last_no_inliers = no_inliers+1

while no_inliers != last_no_inliers:
    last_center = center
    last_dists_mean = dists_mean
    last_no_inliers = no_inliers

    fig = plt.figure(1)
    ax = plt.axes(projection="3d")
    
    ax.scatter3D(coords[inliers,0],coords[inliers,1],coords[inliers,2],s=1)
    
    center = np.mean(coords[inliers],axis=0)
    print("Center: ",center)
    ax.scatter3D(center[0],center[1],center[2],s=5)
    
    dists = np.linalg.norm(coords-center,axis=1)
    dists_mean = np.mean(dists[inliers])
    dists_std = np.std(dists[inliers])
    print("Mean dist: {} Std dev: {}".format(dists_mean, dists_std))
    
    set_axes_equal(ax)
    ax.set_aspect('equal')
    plot_sphere(ax,center,dists_mean + 2*dists_std)
    inliers = np.logical_and(inliers, dists <= ( dists_mean + 2*dists_std ) )
    no_inliers = np.sum(inliers)
    
    print("Points left {}".format(no_inliers))
    #plt.pause(1)
    
    change_in_center = np.linalg.norm(last_center - center)
    change_in_mean = last_dists_mean - dists_mean
    print("Change in center: {} change in mean: {}".format(change_in_center,change_in_mean))
            
    #input("Press any key")
plt.show()

for track_id, is_inlier in zip(sorted(points_dict.keys(),key = int), inliers):
    if not is_inlier:
        points_dict.pop(track_id)

with open(output_recon_file,'w') as f:
    json.dump(data,f)

