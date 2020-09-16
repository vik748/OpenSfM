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
import helper_functions as hf
import numpy as np


log.setup()
logger = logging.getLogger()
#logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)

#image_dataset_location_root = '/data/lowbit_datasets/Stingray2_080718_800x600'
#image_dataset_location_root = '/data/lowbit_datasets/skerki_full'
#image_dataset_location_root = '/data/lowbit_datasets/skerki_mud'
#image_dataset_location_root = '/data/lowbit_datasets/skerki_mud_CLAHE'
#image_dataset_location_root = '/data/lowbit_datasets/ir_day'
image_dataset_location_root = '/data/lowbit_datasets/ir_night'
#image_dataset_location_root = '/data/lowbit_datasets/skerki_full_CLAHE'

image_datasets = ['/data/lowbit_datasets/skerki_full',
                  '/data/lowbit_datasets/skerki_mud',
                  '/data/lowbit_datasets/skerki_mud_CLAHE',
                  '/data/lowbit_datasets/Stingray2_080718_800x600',
                  '/data/lowbit_datasets/ir_day',
                  '/data/lowbit_datasets/ir_night']

def aggregate_dataset_results(image_dataset_location_root):
    processing_location_root = '/data/opensfm_processing'
    open_sfm_install_folder = '/home/auv/apps/OpenSfM'
    
    image_dataset_types = ['RAW', '6_bit', '5_bit', '4_bit', '3_bit']
    feature_detector_types = ['HAHOG', 'ZERNIKE', 'SIFT', 'ORB'] # 'HAHOG','ZERNIKE', , 'ORB', 'SIFT'
    
    image_dataset_name = os.path.basename(image_dataset_location_root)
    image_dataset_folders = [os.path.join(image_dataset_location_root, image_dataset_name+'_' + dt) 
                             for dt in image_dataset_types]
    
    results_output_file = os.path.join(processing_location_root, image_dataset_name, 'results'+time.strftime("_%Y%m%d_%H%M%S")+'.csv')
    
    results_df = pd.DataFrame(columns = ['set_title','n_bits', 'detector', 'descriptor', 'tracks', 
                                         'run', 'cams', 'points', 'max', 'mean', 'median', 'min', 
                                         'rmse', 'sse', 'std'])
    
    nbits_map = {'RAW': 8, '7_bit': 7, '6_bit': 6, '5_bit': 5, '4_bit': 4, '3_bit': 3, '2_bit': 2, '1_bit': 1}
    num_reconstructions = 5
    
    os.chdir(os.path.join(processing_location_root, image_dataset_name))
    
    
    for fd_type in feature_detector_types:
        fd_folder = image_dataset_name + '_' + fd_type
        if not os.path.exists(fd_folder):
            logger.error("Folder: {} does not exist!".format(fd_folder))
            
        traj_plot_cmd = 'evo_traj tum -p -a -s --plot_mode=xy --ref={}'.format('groundtruth_tum.txt')
    
            
        #for id_fold in image_dataset_folders:
        for id_fold, id_type in zip(image_dataset_folders, image_dataset_types):
            result_config_dict = {'set_title': image_dataset_name,
                                  'detector': fd_type,
                                  'descriptor': fd_type,
                                  'n_bits': nbits_map[id_type]}
                    
            id_type_fold = os.path.join(fd_folder, os.path.basename(id_fold))
            
            
            #logger.info("Processing: {0}".format(id_type_fold))
            num_tracks = hf.tracks_results_summary(os.path.join(id_type_fold, "reports", "tracks.json"))
            result_config_dict.update({'tracks': num_tracks})
            
            #if num_tracks > 0:
            for i in range(num_reconstructions):
                result_config_dict.update({'run': i, 'cams': 0, 'points': 0,
                                           'max': np.nan, 'mean': np.nan, 'median': np.nan, 'min': np.nan, 
                                           'rmse': np.nan, 'sse': np.nan, 'std': np.nan})
                curr_run_fold = id_type_fold+"_run{:02d}".format(i)
                recon_file = os.path.join(curr_run_fold, 'reconstruction.json')
                tum_file_name = os.path.join(curr_run_fold, 
                                             fd_type + os.path.basename(id_fold).replace(image_dataset_name,'') + '.txt')
                logger.info("Processing: {0}".format(recon_file))
                                            
                if num_tracks > 0 and os.path.exists(recon_file):
                    _, _, recon_res_dict = hf.recon_results_summary(recon_file)
                    if recon_res_dict['num_models'] > 0:
                        result_config_dict.update({'run': i, 'cams': recon_res_dict.get('model_0_num_cams',0),
                                                   'points': recon_res_dict.get('model_0_num_points',0)})
                        evo_ape_dict = hf.evo_ape_results(curr_run_fold+'/evo_ape/stats.json')
                        result_config_dict.update(evo_ape_dict)
                                            
                    else:
                        logger.error("Recon file: {} 0 models!".format(recon_file))
                        
                else:
                    logger.error("Recon file: {} does not exist!".format(recon_file))
                    
                results_df = results_df.append(result_config_dict, ignore_index=True)
    return results_df
#results_df.to_csv(results_output_file)


results_df = pd.concat([aggregate_dataset_results(ds) for ds in image_datasets], ignore_index=True)
results_output_file = os.path.join(processing_location_root, 'results'+time.strftime("_%Y%m%d_%H%M%S")+'.csv')    
results_df.to_csv(results_output_file)