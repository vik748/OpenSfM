#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 08:11:51 2021

@author: vik748
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import sys,os
import pandas as pd

def tracks_histogram(recon_file, tracks_file, ax, model_num=0, bins=np.linspace(2,15,14)):
    '''
    How the tracks.csv file is written
    template <class S>
    void WriteToStreamCurrentVersion(S& ostream, const TracksManager& manager) {
      ostream << manager.TRACKS_HEADER << "_v" << manager.TRACKS_VERSION
              << std::endl;
      const auto shotsIDs = manager.GetShotIds();
      for (const auto& shotID : shotsIDs) {
        const auto observations = manager.GetShotObservations(shotID);
        for (const auto& observation : observations) {
          ostream << shotID << "\t" << observation.first << "\t"
                  << observation.second.id << "\t" << observation.second.point(0)
                  << "\t" << observation.second.point(1) << "\t"
                  << observation.second.scale << "\t" << observation.second.color(0)
                  << "\t" << observation.second.color(1) << "\t"
                  << observation.second.color(2) << std::endl;
        }
      }
    }

    '''
    with open(recon_file) as f:
        data = json.load(f)

    if model_num == -1:
        points_dict = {}
        for d in data:
            points_dict.update(d['points'])

    else:
        points_dict = data[model_num]['points']

    model_0_point_ids_int = [int(k) for k in points_dict.keys()]

    tracks_df = pd.read_csv(tracks_file, sep='\t', skiprows=1,
                            names=['image', 'track_id', 'feature_id',  'x', 'y',
                                   'scale', 'r', 'g', 'b'])
    track_id_counts = tracks_df.track_id.value_counts()


    model_0_track_id_counts = track_id_counts[model_0_point_ids_int]

    ax.hist(model_0_track_id_counts, bins=bins)


########################################
#   Skerki_mud SIFT - RAW vs CLAHE - Model 0
########################################
recon_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Skerki_mud/Skerki_mud_RAW_SIFT_reconstruction.json'
tracks_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Skerki_mud/Skerki_mud_RAW_SIFT_tracks.csv'

fig1, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
fig1.suptitle('Skerki Mud SIFT')

tracks_histogram(recon_file, tracks_file, ax[0], bins=np.linspace(2,15,14))

ax[0].set_xlim([2, None])
ax[0].set_yscale('log')
ax[0].set_ylim([None, 10000])
ax[0].set_title('RAW')
ax[0].set_xlabel('Feature Track Length')
ax[0].set_ylabel('Fequency')
ax[0].xaxis.set_major_locator(plt.MultipleLocator(1))

recon_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Skerki_mud/Skerki_mud_CLAHE_SIFT_reconstruction.json'
tracks_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Skerki_mud/Skerki_mud_CLAHE_SIFT_tracks.csv'


tracks_histogram(recon_file, tracks_file, ax[1], bins=np.linspace(2,15,14))
ax[1].set_title('CLAHE')
ax[1].set_xlabel('Feature Track Length')
ax[1].set_ylabel('Fequency')


########################################
#   Skerki_mud RAW - SIFT vs Zernike
########################################
recon_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Skerki_mud/Skerki_mud_RAW_SIFT_reconstruction.json'
tracks_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Skerki_mud/Skerki_mud_RAW_SIFT_tracks.csv'

fig2, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
fig2.suptitle('Skerki Mud RAW')

tracks_histogram(recon_file, tracks_file, ax[0], bins=np.linspace(2,15,14))

ax[0].set_xlim([2, None])
ax[0].set_yscale('log')
ax[0].set_ylim([None, 10000])
ax[0].set_title('SIFT')
ax[0].set_xlabel('Feature Track Length')
ax[0].set_ylabel('Fequency')
ax[0].xaxis.set_major_locator(plt.MultipleLocator(1))

recon_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Skerki_mud/Skerki_mud_RAW_ZERNIKE_reconstruction.json'
tracks_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Skerki_mud/Skerki_mud_RAW_ZERNIKE_tracks.csv'


tracks_histogram(recon_file, tracks_file, ax[1], model_num=1, bins=np.linspace(2,15,14))
ax[1].set_title('ZERNIKE')
ax[1].set_xlabel('Feature Track Length')
ax[1].set_ylabel('Fequency')

########################################
#   Skerki_mud Zernike - RAW vs CLAHE
########################################
recon_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Skerki_mud/Skerki_mud_RAW_ZERNIKE_reconstruction.json'
tracks_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Skerki_mud/Skerki_mud_RAW_ZERNIKE_tracks.csv'

fig3, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
fig3.suptitle('Skerki Mud ZERNIKE')

tracks_histogram(recon_file, tracks_file, ax[0], model_num=1, bins=np.linspace(2,15,14))

ax[0].set_xlim([2, None])
ax[0].set_yscale('log')
ax[0].set_ylim([None, 10000])
ax[0].set_title('RAW')
ax[0].set_xlabel('Feature Track Length')
ax[0].set_ylabel('Fequency')
ax[0].xaxis.set_major_locator(plt.MultipleLocator(1))

recon_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Skerki_mud/Skerki_mud_CLAHE_ZERNIKE_reconstruction.json'
tracks_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Skerki_mud/Skerki_mud_CLAHE_ZERNIKE_tracks.csv'


tracks_histogram(recon_file, tracks_file, ax[1], bins=np.linspace(2,15,14))
ax[1].set_title('CLAHE')
ax[1].set_xlabel('Feature Track Length')
ax[1].set_ylabel('Fequency')


########################################
#   Stingray - SIFT vs Zernike
########################################
recon_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Stingray/Stingray_SIFT_reconstruction.json'
tracks_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Stingray/Stingray_SIFT_tracks.csv'

fig4, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
fig4.suptitle('Stingray RAW')

tracks_histogram(recon_file, tracks_file, ax[0], bins=np.linspace(2,15,14))

ax[0].set_xlim([2, None])
ax[0].set_yscale('log')
ax[0].set_ylim([None, 10000])
ax[0].set_title('SIFT')
ax[0].set_xlabel('Feature Track Length')
ax[0].set_ylabel('Fequency')
ax[0].xaxis.set_major_locator(plt.MultipleLocator(1))

recon_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Stingray/Stingray_ZERNIKE_reconstruction.json'
tracks_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Stingray/Stingray_ZERNIKE_tracks.csv'


tracks_histogram(recon_file, tracks_file, ax[1], bins=np.linspace(2,15,14))
ax[1].set_title('ZERNIKE')
ax[1].set_xlabel('Feature Track Length')
ax[1].set_ylabel('Fequency')

########################################
#   Skerki_mud SIFT - RAW vs CLAHE - Combined
########################################
recon_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Skerki_mud/Skerki_mud_RAW_SIFT_reconstruction.json'
tracks_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Skerki_mud/Skerki_mud_RAW_SIFT_tracks.csv'

fig5, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
fig5.suptitle('Skerki Mud SIFT - Combined')

tracks_histogram(recon_file, tracks_file, ax[0], model_num=-1, bins=np.linspace(2,15,14))

ax[0].set_xlim([2, None])
ax[0].set_yscale('log')
ax[0].set_ylim([None, 10000])
ax[0].set_title('RAW')
ax[0].set_xlabel('Feature Track Length')
ax[0].set_ylabel('Fequency')
ax[0].xaxis.set_major_locator(plt.MultipleLocator(1))

recon_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Skerki_mud/Skerki_mud_CLAHE_SIFT_reconstruction.json'
tracks_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Skerki_mud/Skerki_mud_CLAHE_SIFT_tracks.csv'


tracks_histogram(recon_file, tracks_file, ax[1], bins=np.linspace(2,15,14))
ax[1].set_title('CLAHE')
ax[1].set_xlabel('Feature Track Length')
ax[1].set_ylabel('Fequency')