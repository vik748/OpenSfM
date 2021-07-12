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
from helper_functions import save_fig2pdf

def tracks_histogram(recon_file, tracks_file, ax, plot_hist=True, model_num=0, bins=np.linspace(2,15,14)):
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

    if plot_hist:
        ax.hist(model_0_track_id_counts, bins=bins)

    return model_0_track_id_counts

def set_fig_axes(ax):
    ax.legend(loc='upper right')
    ax.set_xlim([2, None])
    #ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.xaxis.set_major_locator(plt.FixedLocator(np.linspace(2,62,13)))
    ax.set_yscale('log')
    ax.set_ylim([0.9, 20000])
    ax.set_xlabel('Feature Track Length',labelpad=0)
    ax.set_ylabel('Frequency',labelpad=-2)
    ax.get_figure().set_size_inches(5,3.5)

########################################
#   Skerki_mud RAW - SIFT vs Zernike
########################################
fig, ax = plt.subplots()
fig.suptitle('Skerki Mud RAW Model 0 vs Model 1')

recon_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Skerki_mud/Skerki_mud_RAW_SIFT_reconstruction.json'
tracks_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Skerki_mud/Skerki_mud_RAW_SIFT_tracks.csv'

counts0 = tracks_histogram(recon_file, tracks_file, ax, plot_hist=False, bins=np.linspace(2,15,14))

recon_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Skerki_mud/Skerki_mud_RAW_ZERNIKE_reconstruction.json'
tracks_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Skerki_mud/Skerki_mud_RAW_ZERNIKE_tracks.csv'

counts1 = tracks_histogram(recon_file, tracks_file, ax, plot_hist=False, model_num=1, bins=np.linspace(2,15,14))

ax.hist([counts0, counts1], np.linspace(2,15,14), label=['SIFT', 'ZERNIKE'])
set_fig_axes(ax)
save_fig2pdf(fig)

########################################
#   Skerki_mud SIFT - RAW vs CLAHE - Model 0
########################################
fig, ax = plt.subplots()
fig.suptitle('Skerki Mud SIFT Model 0 vs Model 0')

recon_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Skerki_mud/Skerki_mud_RAW_SIFT_reconstruction.json'
tracks_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Skerki_mud/Skerki_mud_RAW_SIFT_tracks.csv'

counts0 = tracks_histogram(recon_file, tracks_file, ax, plot_hist=False, bins=np.linspace(2,15,14))

recon_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Skerki_mud/Skerki_mud_CLAHE_SIFT_reconstruction.json'
tracks_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Skerki_mud/Skerki_mud_CLAHE_SIFT_tracks.csv'

counts1 = tracks_histogram(recon_file, tracks_file, ax, plot_hist=False, bins=np.linspace(2,15,14))

ax.hist([counts0, counts1], np.linspace(2,15,14), label=['RAW', 'CLAHE'])
set_fig_axes(ax)
save_fig2pdf(fig)


########################################
#   Skerki_mud SIFT - RAW vs CLAHE - Combined vs Model 0
########################################
fig, ax = plt.subplots()
fig.suptitle('Skerki Mud SIFT Combined vs Model 0')

recon_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Skerki_mud/Skerki_mud_RAW_SIFT_reconstruction.json'
tracks_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Skerki_mud/Skerki_mud_RAW_SIFT_tracks.csv'

counts0 = tracks_histogram(recon_file, tracks_file, ax, model_num=-1, plot_hist=False, bins=np.linspace(2,15,14))

recon_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Skerki_mud/Skerki_mud_CLAHE_SIFT_reconstruction.json'
tracks_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Skerki_mud/Skerki_mud_CLAHE_SIFT_tracks.csv'

counts1 = tracks_histogram(recon_file, tracks_file, ax, plot_hist=False, bins=np.linspace(2,15,14))

ax.hist([counts0, counts1], np.linspace(2,15,14), label=['RAW', 'CLAHE'])
set_fig_axes(ax)
save_fig2pdf(fig)


########################################
#   Skerki_mud Zernike - RAW vs CLAHE Model 1 vs Model 0
########################################

fig, ax = plt.subplots()
fig.suptitle('Skerki Mud Zernike Model 1 vs Model 0')

recon_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Skerki_mud/Skerki_mud_RAW_ZERNIKE_reconstruction_run00.json'
tracks_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Skerki_mud/Skerki_mud_RAW_ZERNIKE_tracks_run00.csv'

counts0 = tracks_histogram(recon_file, tracks_file, ax, model_num=1, plot_hist=False, bins=np.linspace(2,15,14))

recon_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Skerki_mud/Skerki_mud_CLAHE_ZERNIKE_reconstruction_run00.json'
tracks_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Skerki_mud/Skerki_mud_CLAHE_ZERNIKE_tracks_run00.csv'

counts1 = tracks_histogram(recon_file, tracks_file, ax, plot_hist=False, bins=np.linspace(2,15,14))

ax.hist([counts0, counts1], np.linspace(2,15,14), label=['RAW', 'CLAHE'])
set_fig_axes(ax)
save_fig2pdf(fig)


########################################
#   Stingray - SIFT vs Zernike
########################################
fig, ax = plt.subplots()
fig.suptitle('Stingray RAW')

recon_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Stingray/Stingray_SIFT_reconstruction.json'
tracks_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Stingray/Stingray_SIFT_tracks.csv'

counts0 = tracks_histogram(recon_file, tracks_file, ax, plot_hist=False, bins=np.linspace(2,15,14))

recon_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Stingray/Stingray_ZERNIKE_reconstruction.json'
tracks_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Stingray/Stingray_ZERNIKE_tracks.csv'

counts1 = tracks_histogram(recon_file, tracks_file, ax, plot_hist=False, bins=np.linspace(2,15,14))

ax.hist([counts0, counts1], np.linspace(2,40,39), label=['SIFT', 'ZERNIKE'])
set_fig_axes(ax)
save_fig2pdf(fig)


########################################
#   Stingray SIFT - RAW vs CLAHE
########################################
fig, ax = plt.subplots()
fig.suptitle('Stingray SIFT')

recon_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Stingray/Stingray_SIFT_reconstruction.json'
tracks_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Stingray/Stingray_SIFT_tracks.csv'

counts0 = tracks_histogram(recon_file, tracks_file, ax, plot_hist=False, bins=np.linspace(2,15,14))

recon_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Stingray/Stingray_CLAHE_SIFT_reconstruction.json'
tracks_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Stingray/Stingray_CLAHE_SIFT_tracks.csv'

counts1 = tracks_histogram(recon_file, tracks_file, ax, plot_hist=False, bins=np.linspace(2,15,14))

ax.hist([counts0, counts1], np.linspace(2,62,13), label=['RAW', 'CLAHE'])
set_fig_axes(ax)
save_fig2pdf(fig)


########################################
#   Stingray Zernike - RAW vs CLAHE
########################################
fig, ax = plt.subplots()
fig.suptitle('Stingray ZERNIKE')

recon_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Stingray/Stingray_ZERNIKE_reconstruction.json'
tracks_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Stingray/Stingray_ZERNIKE_tracks.csv'

counts0 = tracks_histogram(recon_file, tracks_file, ax, plot_hist=False, bins=np.linspace(2,15,14))

recon_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Stingray/Stingray_CLAHE_ZERNIKE_reconstruction.json'
tracks_file = '/home/vik748/data/OpenSfM_data/track_length_test_data/Stingray/Stingray_CLAHE_ZERNIKE_tracks.csv'

counts1 = tracks_histogram(recon_file, tracks_file, ax, plot_hist=False, bins=np.linspace(2,15,14))

ax.hist([counts0, counts1], np.linspace(2,62,13), label=['RAW', 'CLAHE'])
set_fig_axes(ax)
save_fig2pdf(fig)