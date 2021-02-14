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
import progressbar as pb

log.setup()
logger = logging.getLogger()
#logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)


image_dataset_location_root = '/data/lowbit_datasets/Stingray2_080718_800x600'
#image_dataset_location_root = '/data/lowbit_datasets/skerki_mud'
#image_dataset_location_root = '/data/lowbit_datasets/skerki_mud_CLAHE'
#image_dataset_location_root = '/data/lowbit_datasets/ir_day'
#image_dataset_location_root = '/data/lowbit_datasets/ir_night'
#image_dataset_location_root = '/data/lowbit_datasets/skerki_full'
processing_location_root = '/data/opensfm_processing'
open_sfm_install_folder = '/home/auv/apps/OpenSfM'

#image_dataset_types = ['RAW', '6_bit', '5_bit', '4_bit', '3_bit']
#image_dataset_types = ['5_bit']
#image_dataset_types = ['4_bit', '3_bit']
#feature_detector_types = ['HAHOG', 'ZERNIKE', 'SIFT', 'ORB'] # 'HAHOG','ZERNIKE', , 'ORB', 'SIFT'
#feature_detector_types = ['ORB'] # 'HAHOG','ZERNIKE', , 'ORB', 'SIFT'

run_config = {'image_dataset_location_root': [image_dataset_location_root],
              'processing_location_root' : [processing_location_root],
              'open_sfm_install_folder' : [open_sfm_install_folder],
              'image_depth' : ['RAW', '6_bit', '5_bit', '4_bit', '3_bit'],
              'feature_type' : ['ZERNIKE'], #,,'SIFT','HAHOG', 'ZERNIKE'
              'num_reconstructions': [5] }

param_matrix = { 'feature_min_frames': [600, 1200, 2400, 3600, 4800, 6400], #600, 1200, 2400, 3600, 
                 'lowes_ratio': [0.85, 0.9, 0.92, 0.95], #0.85, 0.9, 
                 'robust_matching_min_match': [10, 12, 15, 17, 20, 25] } # [10, 12, 15, 17] }

image_dataset_name = os.path.basename(image_dataset_location_root)
image_dataset_folders = [os.path.join(image_dataset_location_root, image_dataset_name+'_' + dt) 
                         for dt in run_config['image_depth']]

results_output_file = os.path.join(processing_location_root, image_dataset_name, 'results'+time.strftime("_%Y%m%d_%H%M%S")+'.csv')

assert all([os.path.isdir(fold) for fold in image_dataset_folders]), "Could not find all reuired folders"
#assert all ([config_dict['detector_based_settings'].get(fd_type) is not None for fd_type in feature_detector_types])
        

os.chdir(os.path.join(processing_location_root, image_dataset_name))
assert os.path.isfile('camera_models_overrides.json'), "Could not find 'camera_models_overrides.json'"
assert os.path.isfile('config_template.yaml'), "Could not find 'config_template.yaml'"



results_df = pd.DataFrame()

for rc in pb.progressbar(hf.dict2combinations(run_config),
                         max_value=hf.dict2combinations_length(run_config)):
#    for osfm_config in pb.progressbar(hf.dict2combinations(param_matrix),
#                                      max_value=hf.dict2combinations_length(param_matrix)):
    params_string = ''.join(['_{}_{}'.format(k,v) for k,v in sorted(rc.items())])
    print(params_string)
    curr_results_df = hf.single_run(rc, None)
    results_df = results_df.append(curr_results_df, ignore_index=True)
    results_df.to_csv(results_output_file)