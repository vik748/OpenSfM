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

def aggregate_dataset_results(image_dataset_location_root):
    processing_location_root = '/data/opensfm_processing'
    open_sfm_install_folder = '/home/auv/apps/OpenSfM'
    
    image_dataset_types = ['RAW', '6_bit', '5_bit', '4_bit', '3_bit']
    feature_detector_types = [ 'HAHOG', 'ZERNIKE', 'SIFT', 'ORB'] # 'HAHOG', 'ZERNIKE', 'SIFT', 
    
    image_dataset_name = os.path.basename(image_dataset_location_root)
    image_dataset_folders = [os.path.join(image_dataset_location_root, image_dataset_name+'_' + dt) 
                             for dt in image_dataset_types]
    
    #results_output_file = os.path.join(processing_location_root, image_dataset_name, 'results'+time.strftime("_%Y%m%d_%H%M%S")+'.csv')
    
    results_df = pd.DataFrame(columns = ['set_title','n_bits', 'detector', 'descriptor', 'tracks', 'run', 'cams', 'points', 
                                         'model0_max', 'model0_mean', 'model0_median', 'model0_min', 'model0_rmse', 'model0_sse', 'model0_std', 
                                         'comb_max', 'comb_mean', 'comb_median', 'comb_min', 'comb_rmse', 'comb_sse', 'comb_std'])
    
    nbits_map = {'RAW': 8, '7_bit': 7, '6_bit': 6, '5_bit': 5, '4_bit': 4, '3_bit': 3, '2_bit': 2, '1_bit': 1}
    num_reconstructions = 5
    
    os.chdir(os.path.join(processing_location_root, image_dataset_name))
    
    for fd_type in feature_detector_types:
        fd_folder = image_dataset_name + '_' + fd_type
        if not os.path.exists(fd_folder):
            logger.error("Folder: {} does not exist!".format(fd_folder))
            
        traj_plot_cmd = 'evo_traj tum -a -s --plot_mode=xy --no_warnings --ref={}'.format('groundtruth_tum.txt')
        traj_plot_m0_cmd = traj_plot_cmd + ' --save_plot '+ os.path.join(fd_type+'_traj_plots_m0.png') 
        traj_plot_comb_cmd = traj_plot_cmd + ' --save_plot '+ os.path.join(fd_type+'_traj_plots_comb.png') 
                
        #for id_fold in image_dataset_folders:
        for id_fold, id_type in zip(image_dataset_folders, image_dataset_types):
            result_config_dict = {c:np.nan for c in results_df.columns}
            result_config_dict.update({'set_title': image_dataset_name,
                                       'detector': fd_type,
                                       'descriptor': fd_type,
                                       'n_bits': nbits_map[id_type]})
                    
            id_type_fold = os.path.join(fd_folder, os.path.basename(id_fold))
            
            
            #logger.info("Processing: {0}".format(id_type_fold))
            tracks_file = os.path.join(id_type_fold, "reports", "tracks.json")
            if os.path.exists(tracks_file):
                num_tracks = hf.tracks_results_summary(tracks_file)
            else:
                num_tracks = 0
                
            result_config_dict.update({'tracks': num_tracks})
            
            #if num_tracks > 0:
            for i in range(num_reconstructions):
                result_config_dict.update({'run': i, 'cams': 0, 'points': 0})                
                curr_run_fold = id_type_fold+"_run{:02d}".format(i)
                recon_file = os.path.join(curr_run_fold, 'reconstruction.json')

                tum_file_name = os.path.join(fd_type + os.path.basename(id_fold).replace(image_dataset_name,''))
                logger.info("Processing: {0}".format(recon_file))
                                            
                if num_tracks > 0 and os.path.exists(recon_file):
                    _, _, recon_res_dict = hf.recon_results_summary(recon_file)
                    if recon_res_dict['num_models'] > 0:
                        result_config_dict.update({'run': i, 'cams': recon_res_dict.get('model_0_num_cams',0),
                                                   'points': recon_res_dict.get('model_0_num_points',0)})
                        
                        #evo_ape_dict = hf.evo_ape_results(curr_run_fold+'/evo_ape/stats.json')
                        
                        evo_ape_dict_dump = os.path.join(curr_run_fold,"evo_ape_dict.pkl")
                        if os.path.exists(evo_ape_dict_dump):
                            logger.info("Reloading existing data from: {0}".format(evo_ape_dict_dump))
                            with open(evo_ape_dict_dump, 'rb') as handle:
                                evo_ape_dict = pickle.load(handle)
                        else:
                            evo_ape_dict = hf.align_and_combine_submodels(recon_file, tum_file_prefix=tum_file_name)
                            with open(evo_ape_dict_dump, 'wb') as handle:
                                pickle.dump(evo_ape_dict, handle)                        
                        
                        #print(result_config_dict)
                        if evo_ape_dict is not None:
                            result_config_dict.update(evo_ape_dict)
                            if i == 0:
                                traj_plot_m0_cmd += ' ' + os.path.join(curr_run_fold,tum_file_name)+'_m00.txt'
                                traj_plot_comb_cmd += ' ' + os.path.join(curr_run_fold,tum_file_name)+'_comb.txt'
                            #print(result_config_dict)
                    else:
                        logger.error("Recon file: {} 0 models!".format(recon_file))
                        
                else:
                    logger.error("Recon file: {} does not exist!".format(recon_file))
                    
                results_df = results_df.append(result_config_dict, ignore_index=True)
                
        print(traj_plot_m0_cmd)
        print(traj_plot_comb_cmd)
        '''
        c,o = hf.execute_cmd(traj_plot_m0_cmd, set_conda=True)
        print (o)
        c,o = hf.execute_cmd(traj_plot_comb_cmd, set_conda=True)
        print (o)
        c,o = hf.execute_cmd('for file in *.png; do if [[ "$file" != *_trim.png ]]; then convert "$file" -trim "${file%.png}_trim.png"; fi; done', 
                          set_conda=False)
        print (o)
        '''
        
        with open (fd_type + '_traj_cmd_m0.sh', 'w') as sh_file:
            sh_file.write(traj_plot_m0_cmd + ' -p')
        with open (fd_type + '_traj_cmd_comb.sh', 'w') as sh_file:
            sh_file.write(traj_plot_comb_cmd + ' -p')        
        
    return results_df


image_datasets = ['/data/lowbit_datasets/skerki_full',
                  '/data/lowbit_datasets/skerki_mud',
                  '/data/lowbit_datasets/skerki_mud_CLAHE',
                  '/data/lowbit_datasets/Stingray2_080718_800x600',
                  '/data/lowbit_datasets/ir_day',
                  '/data/lowbit_datasets/ir_night']

#image_datasets = ['/data/lowbit_datasets/skerki_mud_CLAHE']

processing_location_root = '/data/opensfm_processing'

results_output_file = os.path.join(processing_location_root, 'results'+time.strftime("_%Y%m%d_%H%M%S")+'.csv')    

results_df = pd.DataFrame()
for ds in image_datasets:
    results_df = results_df.append(aggregate_dataset_results(ds), ignore_index=True)
    results_df.to_csv(results_output_file)
