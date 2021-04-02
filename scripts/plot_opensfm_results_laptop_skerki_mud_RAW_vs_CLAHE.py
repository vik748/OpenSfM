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
        plt.savefig(os.path.join(folder, fname +'_'+datetime.now().strftime("%Y%m%d%H%M%S") +'.pdf'),format='pdf', dpi=1200,  orientation='landscape', papertype='letter')
    else:
        plt.savefig(fname +'_'+datetime.now().strftime("%Y%m%d%H%M%S") +'.png',format='png', dpi=300)

results_file = '/home/vik748/data/low_contrast_results/results_20200915_005714.csv'

results_df = pd.read_csv(results_file)
#results_df.loc[(results_df['detector'] == 'ORB') && (results_df['descriptor'] == 'ORB')]

groupeddf = results_df.groupby(['set_title','n_bits','detector']).mean()

data_sets = results_df.set_title.unique()[1:3]
#data_sets = ['ir_day']
plot_ys = ['cams', 'points', 'mean', 'rmse']
y_labels = ['No of images registered', r'Avg No of points in Model$_0$', 'Mean Error against groundtruth', 'RMSE Error against groundtruth']
fig,axes = plt.subplots(2,2)
fig.set_size_inches([14, 8])
fig.suptitle(data_sets[0])
colors = ['C0','C1','C2','C3']
line_style = ['-', '--']
markers = ['.', 'x']

for dset, ls, mrk in zip(data_sets, line_style, markers):
    for ax, y_name, ylabel, col in zip(axes.flatten(), plot_ys, y_labels, colors):
        groupeddf.loc[(dset),y_name].unstack().plot(ax=ax, marker=mrk,ms=6, color=colors, linestyle=ls, legend=False)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Bit depth of images')
        ax.set_xlim([8.1, 2.9])
        
handles, labels = ax.get_legend_handles_labels()
newlabels = [l+'_RAW' for l in labels[:4]] + [l+'_CLAHE' for l in labels[:4]]
fig.legend(handles, newlabels, loc='center right')
save_fig2png(fig, size=[14, 8])
