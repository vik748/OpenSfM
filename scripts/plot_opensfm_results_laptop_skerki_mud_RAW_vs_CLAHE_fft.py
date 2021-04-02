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
import matplotlib as mpl
import itertools

def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])


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

def save_fig2pdf(fig, size=[8, 6.7], folder=None, fname=None):
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
        plt.savefig(fname +'_'+datetime.now().strftime("%Y%m%d%H%M%S") +'.pdf',format='pdf', dpi=300)

results_file = '/home/vik748/data/low_contrast_results/results_20200915_005714.csv'

results_df_full = pd.read_csv(results_file)

results_df = results_df_full.loc[results_df_full['detector'] != 'HAHOG']
#results_df.loc[(results_df['detector'] == 'ORB') && (results_df['descriptor'] == 'ORB')]

groupeddf = results_df.groupby(['set_title','n_bits','detector']).mean()

data_sets = results_df.set_title.unique()[1:3]
#data_sets = ['ir_day']
plot_ys = ['cams', 'points']#, 'mean', 'rmse']
y_labels = ['No of images registered', r'Avg No of points in Model$_0$']#, 'Mean Error against groundtruth', 'RMSE Error against groundtruth']
fig,axes = plt.subplots(1,2)
fig.set_size_inches([9, 5])
fig.suptitle(data_sets[0])
colors = ['C0','C1','C2']
line_style = ['-', '--']
markers = ['.', 'x']

for dset, ls, mrk, col in zip(data_sets, line_style, markers, colors):
    for ax, y_name, ylabel in zip(axes.flatten(), plot_ys, y_labels):
        groupeddf.loc[(dset),y_name].unstack().plot(ax=ax, marker=mrk,ms=6, color=colors, linestyle=ls, legend=False)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Bit depth of images')
        ax.set_xlim([8.1, 2.9])

axes[0].set_ylim([0,65])
axes[1].set_ylim([0,12000])

handles, labels = ax.get_legend_handles_labels()
newlabels = [l+'_RAW' for l in labels[:3]] + [l+'_CLAHE' for l in labels[:3]]

fig.subplots_adjust(bottom=0.2,left=0.06, right=0.98,wspace=0.25 )
fig.legend(flip(handles,3), flip(newlabels,3), ncol=3, loc='lower center')
ax.get_yaxis().set_major_formatter(
    mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
#ax.legend(handles,newlabels,loc='center left', bbox_to_anchor=(1, 0.5))
#fig.tight_layout()
save_fig2pdf(fig, size=[9, 5])

#fig2 = plt.figure()
#fig2.legend(handles, newlabels, loc='center')
#save_fig2png(fig)
