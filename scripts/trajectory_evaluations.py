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
processing_location_root = '/data/opensfm_processing'
open_sfm_install_folder = '/home/auv/apps/OpenSfM'

image_dataset_types = ['RAW', '6_bit', '5_bit', '4_bit', '3_bit']
feature_detector_types = ['HAHOG', 'ZERNIKE', 'SIFT', 'ORB'] # 'HAHOG','ZERNIKE', , 'ORB', 'SIFT'

image_dataset_name = os.path.basename(image_dataset_location_root)
image_dataset_folders = [os.path.join(image_dataset_location_root, image_dataset_name+'_' + dt) 
                         for dt in image_dataset_types]

nbits_map = {'RAW': 8, '7_bit': 7, '6_bit': 6, '5_bit': 5, '4_bit': 4, '3_bit': 3, '2_bit': 2, '1_bit': 1}
num_reconstructions = 5

os.chdir(os.path.join(processing_location_root, image_dataset_name))

open('traj_ape_cmd.sh', 'w').close()

for fd_type in feature_detector_types:
    fd_folder = image_dataset_name + '_' + fd_type
    if not os.path.exists(fd_folder):
        logger.error("Folder: {} does not exist!".format(fd_folder))
        
    traj_plot_cmd = 'evo_traj tum -p -a -s --plot_mode=xy --ref={}'.format('groundtruth_tum.txt')

        
    for id_fold in image_dataset_folders:
                
        id_type_fold = os.path.join(fd_folder, os.path.basename(id_fold))
        
        
        #logger.info("Processing: {0}".format(id_type_fold))
        num_tracks = hf.tracks_results_summary(os.path.join(id_type_fold, "reports", "tracks.json"))
        
        if num_tracks > 0:
            for i in range(num_reconstructions):
                curr_run_fold = id_type_fold+"_run{:02d}".format(i)
                recon_file = os.path.join(curr_run_fold, 'reconstruction.json')
                tum_file_name = os.path.join(curr_run_fold, 
                                             fd_type + os.path.basename(id_fold).replace(image_dataset_name,'') + '.txt')
                logger.info("Processing: {0}".format(recon_file))
                                
                if os.path.exists(recon_file):
                    _, _, recon_res_dict = hf.recon_results_summary(recon_file)
                    if recon_res_dict['num_models'] > 0:
                        ret_val = hf.opensfmposes2tum(recon_file, tum_file=tum_file_name, plot_poses=False)
                        # --plot --plot_mode xy 
                        traj_ape_cmd = 'evo_ape tum -vas groundtruth_tum.txt {} --save_results {}/evo_ape.zip || exit 1\n'.format(tum_file_name, curr_run_fold)
                        traj_ape_unzip = 'unzip {}/evo_ape.zip -d {}/evo_ape || exit 1\n '.format(curr_run_fold, curr_run_fold)
                        with open ('traj_ape_cmd.sh', 'a+') as sh_file:
                            sh_file.write(traj_ape_cmd + traj_ape_unzip)
                        if i == 0 and ret_val == 0:
                            traj_plot_cmd += ' ' + tum_file_name
                    else:
                        logger.error("Recon file: {} 0 models!".format(recon_file))
                else:
                    logger.error("Recon file: {} does not exist!".format(recon_file))
                       
    traj_plot_cmd += ' --save_plot ' + os.path.join(fd_type+'_traj_plots.pdf')
        
    with open (fd_type + '_traj_cmd.sh', 'w') as sh_file:
        sh_file.write(traj_plot_cmd)