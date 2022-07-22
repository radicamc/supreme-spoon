#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Jul 21 18:07 2022

@author: MCR

Custom JWST DMS pipeline steps for Stage 4 (lightcurve fitting).
"""

import numpy as np


def do_binning(R, wave, flux, error, full_wave=None):
    if full_wave is None:
        full_wave = wave
    bins = create_wave_axis(R, np.nanmin(full_wave[0]),
                            np.nanmax(full_wave[0]))[1:-1]

    ints, lcs = np.shape(full_wave)
    binned_wave = np.zeros((ints, len(bins) + 1))
    binned_flux = np.zeros((ints, len(bins) + 1))
    binned_error = np.zeros((ints, len(bins) + 1))
    for i in range(len(bins) + 1):
        if i == 0:
            ii = np.where(wave[0] <= bins[i])[0]
            jj = np.where(full_wave[0] <= bins[i])[0]
        elif i == len(bins):
            ii = np.where(wave[0] > bins[i - 1])[0]
            jj = np.where(full_wave[0] > bins[i - 1])[0]
        else:
            ii = np.where((wave[0] <= bins[i]) & (wave[0] > bins[i - 1]))[0]
            jj = \
            np.where((full_wave[0] <= bins[i]) & (full_wave[0] > bins[i - 1]))[
                0]
        binned_wave[:, i] = np.nanmean(full_wave[:, jj], axis=1)
        if len(ii) == 0:
            binned_flux[:, i] = np.nan
            binned_error[:, i] = np.nan
        else:
            binned_flux[:, i] = np.nansum(flux[:, ii], axis=1)
            binned_error[:, i] = np.sqrt(np.nanmean(error[:, ii] ** 2, axis=1))

    return binned_wave, binned_flux, binned_error


def create_wave_axis(R, wave_min, wave_max):
    ww = wave_min
    wave_ax = []
    while ww <= wave_max:
        wave_ax.append(ww)
        step = ww / R
        ww += step
    wave_ax.append(wave_max)

    return wave_ax


def run_stage4():
    return

# TODO: Add main to just run stage 4
if __name__ == "__main__":
    run_stage4()
