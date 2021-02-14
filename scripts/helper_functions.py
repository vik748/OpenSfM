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
from itertools import product

def execute_cmd(command, set_conda=False, cwd=None):
    if type(command) is list:
        command = ' '.join(command)
    if set_conda:
        command = 'source ~/conda_init && conda activate simple_vslam_env && ' + command
    with Popen(command, stdout=PIPE, stderr=PIPE, bufsize=1, 
               shell=True, universal_newlines=True, executable='bash', cwd=cwd) as p:
        output = ""
        for line in p.stderr:
            #print("\r\t>>> "+line, end='') # process line here
            output+=line
        for line in p.stdout:
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

def opensfmshots2tum(shots, model_tum_file='reconstruction_model.txt', plot_poses = False):
    '''
    This function converts opensfm reconstruction json file
    to tum format for evo and rpg_trajector_evaluation packages as 'time x y z qx qy qz qw'
    '''    
    if plot_poses:        
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect('equal')         # important!
        title = ax.set_title('3D Test')
        
    if len(shots) > 2:            
                           
        output_list = []
        time_list = []
        for cam_name, shot in sorted(shots.items()):
            axis_angle_world = np.array(shot['rotation'])
            t_world = np.array(shot['translation'])
            
            R, t = world_axangle_t_2_object_poseRT(axis_angle_world, t_world)
            
            qw, qx, qy, qz = mat2quat(R)
            i_time = camname2time(os.path.splitext(cam_name)[0])
            output_list.append([*t[:,0].tolist(), qx, qy, qz, qw])
            time_list.append(i_time)
            
            if plot_poses:
                plot_pose3RT_on_axes(ax, R, t.T, axis_length=.5)
                set_axes_equal(ax)
                plt.pause(.1)
                
        df = pd.DataFrame(output_list, columns = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'], index=time_list)
        df.to_csv(model_tum_file, sep=' ', header=False)
        print("DF written to ",model_tum_file)
        return 0
    else: 
        return 1

def align_and_combine_submodels(opensfm_json, tum_file_prefix=None, plot_poses = False):
    '''
    This function opensfm reconstruction submodels into text files, then runs evo alignment
    and scaling and combines the resulting trajectories into a single file
    in tum format as 'time x y z qx qy qz qw'
    '''    
    gt_file = '../../groundtruth_tum.txt'
    recon_dir = os.path.dirname(opensfm_json)
    
    if tum_file_prefix is None:
        combined_tum_file = os.path.splitext(opensfm_json)[0] +'_tum_combined.txt'
        tum_file = os.path.splitext(opensfm_json)[0] +'_tum.txt'
    else:
        combined_tum_file = tum_file_prefix +'_comb.txt'
        tum_file = tum_file_prefix +'.txt'

    with open(opensfm_json) as f:
        data = json.load(f)
        
    if len(data) > 0 and len(data[0]['shots']) > 2 :
        evo_traj_all_models_cmd = 'evo_traj tum -a -s --plot_mode=xy --save_plot submodel_trajs.png --no_warnings --ref={}'.format(gt_file)
        aligned_trajs = []
        for model_num, model in enumerate(data):
            shots = model['shots']
            if len(shots) > 2:            
                name, ext = os.path.splitext(tum_file)
                model_tum_file = os.path.join(recon_dir, '{}_m{:02d}{}'.format(name, model_num, ext))
                opensfmshots2tum(shots, model_tum_file=model_tum_file, plot_poses = False)
                
                evo_traj_model_cmd = 'evo_traj tum -a -s --ref={} {} --save_as_tum --no_warnings'.format(
                                gt_file, os.path.basename(model_tum_file))
                evo_traj_all_models_cmd += ' ' + os.path.basename(model_tum_file)
                
                c,o = execute_cmd(evo_traj_model_cmd, set_conda=True, cwd=recon_dir)
                print (o)
                aligned_trajs.append(pd.read_csv(os.path.splitext(model_tum_file)[0]+".tum", 
                                                 sep=' ', names=['time', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'], 
                                                 index_col='time'))               
                
        
        combined_traj_df = pd.concat(aligned_trajs).sort_index()
        combined_traj_df.to_csv(os.path.join(recon_dir, combined_tum_file), sep=' ', header=False)
        
        print (evo_traj_all_models_cmd)
        c,o = execute_cmd(evo_traj_all_models_cmd, set_conda=True, cwd=recon_dir)
        print (o)
        c,o = execute_cmd('for file in *.png; do if [[ "$file" != *_trim.png ]]; then convert "$file" -trim "${file%.png}_trim.png"; fi; done', 
                          set_conda=False, cwd=recon_dir)
        
        # RUN analysis on model 0 trajectory
        model0_tum_file = '{}_m{:02d}{}'.format(name, 0, ext)
        traj_ape_model0_cmd = 'evo_ape tum -vas {} {} --no_warnings --save_results evo_ape_model0.zip'.format(gt_file, model0_tum_file)
        traj_ape_model0_unzip = 'unzip -o evo_ape_model0.zip -d evo_ape_model0'
        c,o = execute_cmd(traj_ape_model0_cmd, set_conda=True, cwd=recon_dir)
        print (o)
        c,o = execute_cmd(traj_ape_model0_unzip, set_conda=True, cwd=recon_dir)
        print (o)
        evo_ape_results_dict = {'model0_'+k: v for k, v in evo_ape_results(recon_dir+'/evo_ape_model0/stats.json').items()} 
        
        # RUN analysis on combined trajectory
        traj_ape_comb_cmd = 'evo_ape tum -vas {} {} --no_warnings --save_results evo_ape_comb.zip'.format(gt_file, combined_tum_file)
        traj_ape_comb_unzip = 'unzip -o evo_ape_comb.zip -d evo_ape_comb'
        c,o = execute_cmd(traj_ape_comb_cmd, set_conda=True, cwd=recon_dir)
        print (o)
        c,o = execute_cmd(traj_ape_comb_unzip, set_conda=True, cwd=recon_dir)
        print (o)
        evo_ape_results_dict.update({'comb_'+k: v for k, v in evo_ape_results(recon_dir+'/evo_ape_comb/stats.json').items()})

        return evo_ape_results_dict
    else:
        return None
        
def single_run(run_config, opensfm_config = None):
    '''
    Perform a single run where all the run settings are specified in the run_config dict 
    
    ex image_dataset_structure:
    └── image_dataset_location_root
        ├── image_depth_source
        ├── image_depth_source
        ├── image_depth_source
        ├── image_depth_source
        ├── image_depth_source
        └── image_depth_source

    starting processing structure:
    └── processing_location_root
        └── 
            ├── camera_models_overrides.json
            └── config_template.yaml
    
    resulting output structure
    └── processing_location_root
        └── image_dataset_name
            ├── camera_models_overrides.json
            └── config_template.yaml
                └── image_depth_target
                └── image_depth_target
                └── image_depth_target
                └── image_depth_target
                └── image_depth_target
                └── image_depth_target
    '''
    # Switch to folder dataset_name folder under processing location
    log.setup()
    logger = logging.getLogger()
    
    results_df = pd.DataFrame(columns = ['set_title','n_bits', 'detector', 'descriptor', 'tracks', 
                                         'run', 'cams', 'points' ])

    nbits_map = {'RAW': 8, '7_bit': 7, '6_bit': 6, '5_bit': 5, '4_bit': 4, '3_bit': 3, '2_bit': 2, '1_bit': 1}
    
    image_dataset_name = os.path.basename(run_config['image_dataset_location_root'])
    
    result_config_dict = {'set_title': image_dataset_name,
                          'detector': run_config['feature_type'],
                          'descriptor': run_config['feature_type'],
                          'n_bits': nbits_map[run_config['image_depth']]}
        
    os.chdir(os.path.join(run_config['processing_location_root'], image_dataset_name))
    
    # Create feature detector level folder    
    fd_folder = image_dataset_name + '_' + run_config['feature_type']
    if not os.path.exists(fd_folder):
        os.makedirs(fd_folder)
    else:
        logger.info("Folder: {} already exists!".format(fd_folder))

    image_depth_source = os.path.join(run_config['image_dataset_location_root'], 
                                      image_dataset_name+'_' + run_config['image_depth'])
     
    image_depth_target = os.path.join(fd_folder, image_dataset_name+'_' + run_config['image_depth'])
    
    if opensfm_config is not None:
        image_depth_target += ''.join(['_{}_{}'.format(k,v) for k,v in sorted(opensfm_config.items())])
    
    logger.info("Processing: {0}".format(run_config))
    
    if not os.path.exists(image_depth_target):
        os.makedirs(image_depth_target)
    else:
        logger.info("Folder: {} already exists!".format(image_depth_target))
    
    try:
        os.symlink(image_depth_source, os.path.join(image_depth_target,'images'))
        copy('camera_models_overrides.json', image_depth_target)
    except FileExistsError as err:
        logger.info("FileExistsError error: {0}".format(err))    
    
    with open(r'config_template.yaml') as file:
        config_dict = yaml.load(file)
        
    detector_based_setting_dict = config_dict.pop('detector_based_settings').get(run_config['feature_type'])
    config_dict.update(detector_based_setting_dict)
    
    config_dict['feature_type'] = run_config['feature_type']
    
    if opensfm_config is not None:
        config_dict.update(opensfm_config)
    
    with open(os.path.join(image_depth_target,'config.yaml'), 'w') as file:
        yaml_output = yaml.dump(config_dict, file, default_flow_style=False)
    
    opensfm_cmds = ["extract_metadata", "detect_features", "match_features", 
                    "create_tracks"]
    #opensfm_cmds = ["reconstruct"]
    
    print(image_depth_target + ' : ',end='',flush=True)
    try:
        if not os.path.exists(image_depth_target+"_before_reconstruct"):
            for osfmcmd in opensfm_cmds:
                print(osfmcmd + ' -> ',end='',flush=True)
                command = [os.path.join(run_config['open_sfm_install_folder'], 'bin/opensfm'), 
                           osfmcmd, image_depth_target]
                #print(command)
                c,o = execute_cmd(command)            
                #print(o)
            copytree(image_depth_target, image_depth_target+"_before_reconstruct", symlinks=True)
        
        else:
            logger.info("Folder: {} already exists, reusing matches!".format(image_depth_target+"_before_reconstruct"))
        
        num_tracks = tracks_results_summary(os.path.join(image_depth_target, "reports", "tracks.json"))
        result_config_dict.update({'tracks': num_tracks})
        
        for i in range(run_config['num_reconstructions']):
            result_config_dict.update({'run': i, 'cams': 0, 'points': 0})
            if num_tracks > 0:
                curr_run_fold = image_depth_target+"_run{:02d}".format(i)
                if os.path.exists(curr_run_fold):
                    move(curr_run_fold, curr_run_fold+'_old_'+time.strftime("%Y%m%d_%H%M%S"))
                copytree(image_depth_target, curr_run_fold, symlinks=True)
                command = [os.path.join(run_config['open_sfm_install_folder'], 'bin/opensfm'), 
                           'reconstruct', curr_run_fold]
                execute_cmd(command)
                results, model_image_names, recon_res_dict = recon_results_summary(os.path.join(curr_run_fold, "reconstruction.json"))
                logger.info("Reconstruction run {:02d}".format(i))
                logger.info(results)
                result_config_dict.update({'run': i, 'cams': recon_res_dict.get('model_0_num_cams',0),
                                           'points': recon_res_dict.get('model_0_num_points',0)})
                if opensfm_config is not None:
                    result_config_dict.update(opensfm_config)
                
            results_df = results_df.append(result_config_dict, ignore_index=True)
        
    except CalledProcessError as err:
        logger.warning("{0} Failed!!! \nCalledProcessError error: {1}".format(image_depth_target,err))

    return results_df

def dict2combinations (dict_in):
    '''
    Given a dictionary where each key points to a list return an iterator which 
    returns dicts with a combination from each list. eg. 
        dict_in = {"num": [1, 2, 3],
                   "letter": ['a', 'b', 'c'],
                   "an":'c'}
        returns:    {'num': 1, 'letter': 'a', 'an': 'c'}
                    {'num': 1, 'letter': 'b', 'an': 'c'}
                    {'num': 1, 'letter': 'c', 'an': 'c'}
                    {'num': 2, 'letter': 'a', 'an': 'c'}
                    {'num': 2, 'letter': 'b', 'an': 'c'}
                    {'num': 2, 'letter': 'c', 'an': 'c'}
                    {'num': 3, 'letter': 'a', 'an': 'c'}
                    {'num': 3, 'letter': 'b', 'an': 'c'}
                    {'num': 3, 'letter': 'c', 'an': 'c'}    
    '''
    for combination in product(*dict_in.values()):
        d = dict(zip(dict_in.keys(), combination))
        yield d
        
def dict2combinations_length(dict_in):
    '''
    Given a dictionary where each key points to list, return the possibile combinations
    by multiplying the lengths of each list
    '''
    prod_len = 1
    for v in dict_in.values():    
        if type(v) == list:
            prod_len *= len(v)
    return prod_len