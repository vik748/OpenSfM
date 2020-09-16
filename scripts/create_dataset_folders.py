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
from shutil import copy, copytree, move
import yaml
#import subprocess #import PIPE, run, STDOUT        
from subprocess import Popen, PIPE, CalledProcessError
from opensfm import log
import json
import pandas as pd
import time

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

def tracks_results_summary(recon_file):
    with open(recon_file) as f:
        data = json.load(f)
        
    return data['num_tracks']

log.setup()
logger = logging.getLogger()
#logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)


#image_dataset_location_root = '/data/lowbit_datasets/Stingray2_080718_800x600'
image_dataset_location_root = '/data/lowbit_datasets/skerki_mud_CLAHE'
#image_dataset_location_root = '/data/lowbit_datasets/ir_night'
#image_dataset_location_root = '/data/lowbit_datasets/skerki_full_CLAHE'
processing_location_root = '/data/opensfm_processing'
open_sfm_install_folder = '/home/auv/apps/OpenSfM'

image_dataset_types = ['RAW', '6_bit', '5_bit', '4_bit', '3_bit']
#image_dataset_types = ['5_bit']
#image_dataset_types = ['4_bit', '3_bit']
feature_detector_types = ['HAHOG', 'ZERNIKE', 'SIFT', 'ORB'] # 'HAHOG','ZERNIKE', , 'ORB', 'SIFT'
#feature_detector_types = ['ORB'] # 'HAHOG','ZERNIKE', , 'ORB', 'SIFT'

image_dataset_name = os.path.basename(image_dataset_location_root)
image_dataset_folders = [os.path.join(image_dataset_location_root, image_dataset_name+'_' + dt) 
                         for dt in image_dataset_types]

results_output_file = os.path.join(processing_location_root, image_dataset_name, 'results'+time.strftime("_%Y%m%d_%H%M%S")+'.csv')

results_df = pd.DataFrame(columns = ['set_title','n_bits', 'detector', 'descriptor', 'tracks', 
                                     'run_0_cams', 'run_0_points', 'run_1_cams', 'run_1_points', 
                                     'run_2_cams', 'run_2_points', 'run_3_cams', 'run_3_points',
                                     'run_4_cams', 'run_4_points'])

nbits_map = {'RAW': 8, '7_bit': 7, '6_bit': 6, '5_bit': 5, '4_bit': 4, '3_bit': 3, '2_bit': 2, '1_bit': 1}
num_reconstructions = 5

assert all([os.path.isdir(fold) for fold in image_dataset_folders]), "Could not find all reuired folders"
#assert all ([config_dict['detector_based_settings'].get(fd_type) is not None for fd_type in feature_detector_types])
        

os.chdir(os.path.join(processing_location_root, image_dataset_name))
assert os.path.isfile('camera_models_overrides.json'), "Could not find 'camera_models_overrides.json'"
assert os.path.isfile('config_template.yaml'), "Could not find 'config_template.yaml'"

for fd_type in feature_detector_types:
    fd_folder = image_dataset_name + '_' + fd_type
    if not os.path.exists(fd_folder):
        os.makedirs(fd_folder)
    else:
        logger.info("Folder: {} already exists!".format(fd_folder))
        
    for id_fold, id_type in zip(image_dataset_folders, image_dataset_types):
        result_config_dict = {'set_title': image_dataset_name,
                              'detector': fd_type,
                              'descriptor': fd_type,
                              'n_bits': nbits_map[id_type]}
                
        id_type_fold = os.path.join(fd_folder, os.path.basename(id_fold))
        logger.info("Processing: {0}".format(id_type_fold))
        if not os.path.exists(id_type_fold):
            os.makedirs(id_type_fold)
        else:
            logger.info("Folder: {} already exists!".format(id_type_fold))
        
        try:
            os.symlink(id_fold, os.path.join(id_type_fold,'images'))            
        except FileExistsError as err:
            logger.info("FileExistsError error: {0}".format(err))

        
        try:
            copy('camera_models_overrides.json', id_type_fold)
        except FileExistsError as err:
            logger.info("FileExistsError error: {0}".format(err))
    
        
        with open(r'config_template.yaml') as file:
            config_dict = yaml.load(file)
            
        detector_based_setting_dict = config_dict.pop('detector_based_settings').get(fd_type)
        config_dict.update(detector_based_setting_dict)
        
        config_dict['feature_type'] = fd_type
        
        with open(os.path.join(id_type_fold,'config.yaml'), 'w') as file:
            yaml_output = yaml.dump(config_dict, file, default_flow_style=False)
        
        opensfm_cmds = ["extract_metadata", "detect_features", "match_features", 
                        "create_tracks"]
        #opensfm_cmds = ["reconstruct"]
        
        print(id_type_fold + ' : ',end='',flush=True)
        try:
            if not os.path.exists(id_type_fold+"_before_reconstruct"):
                for osfmcmd in opensfm_cmds:
                    print(osfmcmd + ' -> ',end='',flush=True)
                    command = [os.path.join(open_sfm_install_folder, 'bin/opensfm'), 
                               osfmcmd, id_type_fold]
                    execute_cmd(command)            
                copytree(id_type_fold, id_type_fold+"_before_reconstruct", symlinks=True)
            
            else:
                logger.info("Folder: {} already exists, reusing matches!".format(id_type_fold+"_before_reconstruct"))
            
            num_tracks = tracks_results_summary(os.path.join(id_type_fold, "reports", "tracks.json"))
            result_config_dict.update({'tracks': num_tracks})
            
            if num_tracks > 0:
                for i in range(num_reconstructions):                    
                    curr_run_fold = id_type_fold+"_run{:02d}".format(i)
                    if os.path.exists(curr_run_fold):
                        move(curr_run_fold, curr_run_fold+'_old_'+time.strftime("%Y%m%d_%H%M%S"))
                    copytree(id_type_fold, curr_run_fold, symlinks=True)
                    command = [os.path.join(open_sfm_install_folder, 'bin/opensfm'), 
                               'reconstruct', curr_run_fold]
                    execute_cmd(command)
                    results, model_image_names, recon_res_dict = recon_results_summary(os.path.join(curr_run_fold, "reconstruction.json"))
                    logger.info("Reconstruction run {:02d}".format(i))
                    logger.info(results)
                    result_config_dict.update({'run_{:d}_cams'.format(i) : recon_res_dict.get('model_0_num_cams',0),
                                               'run_{:d}_points'.format(i) : recon_res_dict.get('model_0_num_points',0)})
            else:
                logger.info("Num tracks = 0, no reconstructions")
                for i in range(num_reconstructions):
                    result_config_dict.update({'run_{:d}_cams'.format(i) : 0,
                                               'run_{:d}_points'.format(i) : 0})
            
            results_df = results_df.append(result_config_dict, ignore_index=True)
            #logger.debug(model_image_names)
        except CalledProcessError as err:
            logger.warning("{0} Failed!!! \nCalledProcessError error: {1}".format(id_type_fold,err))
            
        results_df.to_csv(results_output_file)