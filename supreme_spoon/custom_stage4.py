#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Jul 21 18:07 2022

@author: MCR

Custom JWST DMS pipeline steps for Stage 4 (lightcurve fitting).
"""

# TODO: input NaNs in fit results for skipping bins so that the fit results arrays are same lengths as wave bins
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


def bin_at_resolution(wavelengths, depths, R=100):
    """FROM Néstor Espinoza
    Function that bins input wavelengths and transit depths (or any other observable, like flux) to a given
    resolution `R`. Useful for binning transit depths down to a target resolution on a transit spectrum.
    Parameters
    ----------
    wavelengths : np.array
        Array of wavelengths

    depths : np.array
        Array of depths at each wavelength.
    R : int
        Target resolution at which to bin (default is 100)
    method : string
        'mean' will calculate resolution via the mean --- 'median' via the median resolution of all points
        in a bin.
    Returns
    -------
    wout : np.array
        Wavelength of the given bin at resolution R.
    dout : np.array
        Depth of the bin.
    derrout : np.array
        Error on depth of the bin.

    """

    # Sort wavelengths from lowest to highest:
    idx = np.argsort(wavelengths)

    ww = wavelengths[idx]
    dd = depths[idx]

    # Prepare output arrays:
    wout, dout, derrout = np.array([]), np.array([]), np.array([])

    oncall = False

    # Loop over all (ordered) wavelengths:
    for i in range(len(ww)):

        if not oncall:

            # If we are in a given bin, initialize it:
            current_wavs = np.array([ww[i]])
            current_depths = np.array(dd[i])
            oncall = True

        else:

            # On a given bin, append next wavelength/depth:
            current_wavs = np.append(current_wavs, ww[i])
            current_depths = np.append(current_depths, dd[i])

            # Calculate current mean R:
            current_R = np.mean(current_wavs) / np.abs(
                current_wavs[0] - current_wavs[-1])

            # If the current set of wavs/depths is below or at the target resolution, stop and move to next bin:
            if current_R <= R:
                wout = np.append(wout, np.mean(current_wavs))
                dout = np.append(dout, np.mean(current_depths))
                derrout = np.append(derrout,
                                    np.sqrt(np.var(current_depths)) / np.sqrt(
                                        len(current_depths)))

                oncall = False

    lw = np.concatenate([wout[:, None], np.roll(wout, 1)[:, None]], axis=1)
    up = np.concatenate([wout[:, None], np.roll(wout, -1)[:, None]], axis=1)

    uperr = (np.mean(up, axis=1) - wout)[:-1]
    uperr = np.insert(uperr, -1, uperr[-1])

    lwerr = (wout - np.mean(lw, axis=1))[1:]
    lwerr = np.insert(lwerr, 0, lwerr[0])

    werrout = [lwerr, uperr]

    return wout, werrout, dout, derrout


def save_transmissions_spectrum(fitres_o1, fitres_o2, output_dir):
    return


def run_stage4():
    return

# TODO: Add main to just run stage 4
if __name__ == "__main__":
    run_stage4()
