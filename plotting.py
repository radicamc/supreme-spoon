#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 14:02 2022

@author: MCR

JWST data analysis plotting routines.
"""

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import warnings


def plot_2dlightcurves(flux1, flux2, wave1, wave2, savename=None):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')

        fig = plt.figure(figsize=(12, 5), facecolor='white')
        gs = GridSpec(1, 2, width_ratios=[1, 1])

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(flux1.T, aspect='auto', origin='lower',
                   vmax=np.nanpercentile(flux1, 95),
                   vmin=np.nanpercentile(flux1, 5),
                   extent=(0, flux1.shape[0]-1, wave1[0], wave1[-1]))
        ax1.set_ylabel('Wavelength [µm]', fontsize=16)
        ax1.set_xlabel('Integration Number', fontsize=16)
        plt.title('Order 1', fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        ax2 = fig.add_subplot(gs[0, 1])
        pp = ax2.imshow(flux2.T, aspect='auto', origin='lower',
                        vmax=np.nanpercentile(flux1, 95),
                        vmin=np.nanpercentile(flux1, 5),
                        extent=(0, flux2.shape[0]-1, wave2[0], wave2[-1]))
        cax = ax2.inset_axes([1.05, 0.005, 0.03, 0.99],
                             transform=ax2.transAxes)
        cb = fig.colorbar(pp, ax=ax2, cax=cax)
        cb.set_label('Normalized Flux', labelpad=15, rotation=270, fontsize=16)
        ax2.set_xlabel('Integration Number', fontsize=16)
        plt.title('Order 2', fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        gs.update(wspace=0.1)

    if savename is not None:
        plt.savefig(savename, format='pdf')
    plt.show()

