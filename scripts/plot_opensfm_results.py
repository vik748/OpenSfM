#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 20:06:06 2019

@author: vik748
"""
import numpy as np
import sys
from matplotlib import pyplot as plt
import os
import pandas as pd
import scipy.stats as st
import re
from datetime import datetime

def save_fig2png(fig, size=[8, 6.7], folder=None, fname=None):    
    if size is None:
        fig.set_size_inches([8, 6.7])
    else:
        fig.set_size_inches(size)
    plt.pause(.1)
    if fname is None:
        if fig._suptitle is None:
            fname = 'figure_{:d}'.format(fig.number)
        else:
            ttl = fig._suptitle.get_text()
            ttl = ttl.replace('$','').replace('\n','_').replace(' ','_')
            fname = re.sub(r"\_\_+", "_", ttl) 
    if folder:
        fig.savefig(os.path.join(folder, fname +'_'+datetime.now().strftime("%Y%m%d%H%M%S") +'.pdf'),format='pdf', dpi=1200,  orientation='landscape', papertype='letter')
    else:
        fig.savefig(fname +'_'+datetime.now().strftime("%Y%m%d%H%M%S") +'.png',format='png', dpi=300)

#results_file = '/home/vik748/data/low_contrast_results/results_20200915_005714.csv'
#results_file = '/home/vik748/data/low_contrast_results/New_liner_low_bitrate/results_20200928_172249 - Full set of results low bitrate.csv'
results_file = '/home/vik748/data/low_contrast_results/New_liner_low_bitrate/results_20200929_235607 - Full set after fixing issues with Stingray and mud CLAHE.csv'

results_df = pd.read_csv(results_file)
#results_df.loc[(results_df['detector'] == 'ORB') && (results_df['descriptor'] == 'ORB')]

groupeddf = results_df.groupby(['set_title','n_bits','detector']).mean()

data_sets = results_df.set_title.unique()
#data_sets = ['ir_day']

#plot_ys = ['cams', 'points', 'comb_mean', 'comb_rmse']
plot_ys = ['cams', 'points', 'model0_mean', 'model0_rmse']

for dset in data_sets:
    fig,axes = plt.subplots(2,2)
    fig.suptitle(dset+"_model0")
    
    for ax, y_name in zip(axes.flatten(), plot_ys):
        groupeddf.loc[(dset),y_name].unstack().plot(ax=ax, marker='.',ms=8)
        ax.set_ylabel(y_name)
        ax.invert_xaxis()
    save_fig2png(fig, size=[16.0, 6])
