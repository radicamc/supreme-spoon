#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Jul 21 18:07 2022

@author: MCR

Custom JWST DMS pipeline steps for Stage 4 (lightcurve fitting).
"""

from exotic_ld import StellarLimbDarkening
import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import numpy as np

from jwst import datamodels
from jwst.pipeline import calwebb_spec2


def bin_at_resolution(wavelengths, depths, depth_error, R=100):
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
    de = depth_error[idx]

    # Prepare output arrays:
    wout, dout, derrout = np.array([]), np.array([]), np.array([])

    oncall = False

    # Loop over all (ordered) wavelengths:
    for i in range(len(ww)):

        if not oncall:

            # If we are in a given bin, initialize it:
            current_wavs = np.array([ww[i]])
            current_depths = np.array(dd[i])
            current_errors = np.array(de[i])
            oncall = True

        else:

            # On a given bin, append next wavelength/depth:
            current_wavs = np.append(current_wavs, ww[i])
            current_errors = np.append(current_errors, de[i])
            current_depths = np.append(current_depths, dd[i])

            # Calculate current mean R:
            current_R = np.mean(current_wavs) / np.abs(current_wavs[0] - current_wavs[-1])

            # If the current set of wavs/depths is below or at the target resolution, stop and move to next bin:
            if current_R <= R:
                wout = np.append(wout, np.nanmean(current_wavs))
                dout = np.append(dout, np.nanmean(current_depths))
                derrout = np.append(derrout, np.sqrt(np.nansum(current_errors**2)/len(current_errors)))

                oncall = False

    lw = np.concatenate([wout[:, None], np.roll(wout, 1)[:, None]], axis=1)
    up = np.concatenate([wout[:, None], np.roll(wout, -1)[:, None]], axis=1)

    uperr = (np.mean(up, axis=1) - wout)[:-1]
    uperr = np.insert(uperr, -1, uperr[-1])

    lwerr = (wout - np.mean(lw, axis=1))[1:]
    lwerr = np.insert(lwerr, 0, lwerr[0])

    werrout = [lwerr, uperr]

    return wout, werrout, dout, derrout


def bin_2d_spectra(wave2d, flux2d, R=150):
    nints, nwave = np.shape(wave2d)

    for i in tqdm(range(nints)):
        if i == 0:
            wc_bin, we_bin, f_bin, e_bin = bin_at_resolution(wave2d[i], flux2d[i], R=R)
            wl_bin, wu_bin = we_bin
        elif i == 1:
            wc_bin_i, we_bin_i, f_bin_i, e_bin_i = bin_at_resolution(wave2d[i], flux2d[i], R=R)
            wl_bin_i, wu_bin_i = we_bin_i
            wc_bin = np.stack([wc_bin, wc_bin_i])
            wl_bin = np.stack([wl_bin, wl_bin_i])
            wu_bin = np.stack([wu_bin, wu_bin_i])
            f_bin = np.stack([f_bin, f_bin_i])
            e_bin = np.stack([e_bin, e_bin_i])
        else:
            wc_bin_i, we_bin_i, f_bin_i, e_bin_i = bin_at_resolution(wave2d[i], flux2d[i], R=R)
            wl_bin_i, wu_bin_i = we_bin_i
            wc_bin = np.concatenate([wc_bin, wc_bin_i[None, :]], axis=0)
            wl_bin = np.concatenate([wl_bin, wl_bin_i[None, :]], axis=0)
            wu_bin = np.concatenate([wu_bin, wu_bin_i[None, :]], axis=0)
            f_bin = np.concatenate([f_bin, f_bin_i[None, :]], axis=0)
            e_bin = np.concatenate([e_bin, e_bin_i[None, :]], axis=0)

    return wc_bin, wl_bin, wu_bin, f_bin, e_bin


def save_transmission_spectrum(wave, wave_err, dppm, dppm_err, order, outdir,
                               filename, target, extraction_type):
    dd = {'wave': wave,
          'wave_err': wave_err,
          'dppm': dppm,
          'dppm_err': dppm_err,
          'order': order}
    df = pd.DataFrame(data=dd)
    if os.path.exists(outdir + filename):
        os.remove(outdir + filename)
    f = open(outdir + filename, 'a')
    f.write('# Target: {}\n'.format(target))
    f.write('# Instrument: NIRISS/SOSS\n')
    f.write('# Pipeline: Supreme-SPOON\n')
    f.write('# 1D Extraction : {}\n'.format(extraction_type))
    f.write('# Author: {}\n'.format(os.environ.get('USER')))
    f.write('# Date: {}\n'.format(datetime.utcnow().replace(microsecond=0).isoformat()))
    f.write('# Column wave: Central wavelength of bin (micron)\n')
    f.write('# Column wave_err: Wavelength bin halfwidth (micron)\n')
    f.write('# Column dppm: (Rp/R*)^2 (ppm)\n')
    f.write('# Column dppm_err: Error in (Rp/R*)^2 (ppm)\n')
    f.write('# Column order: SOSS diffraction order\n')
    f.write('#\n')
    df.to_csv(f, index=False)
    f.close()


def gen_ld_coefs(datafile, wavebin_low, wavebin_up, order, M_H, logg, Teff):
    ld_data_path = '/home/radica/.anaconda3/envs/atoca/lib/python3.10/site-packages/exotic_ld/exotic-ld_data/'
    sld = StellarLimbDarkening(M_H, Teff, logg, '1D', ld_data_path)
    mode = 'custom'

    step = calwebb_spec2.extract_1d_step.Extract1dStep()
    spectrace_ref = step.get_reference_file(datafile, 'spectrace')
    spec_trace = datamodels.SpecTraceModel(spectrace_ref)
    wavelengths = spec_trace.trace[order].data['WAVELENGTH']*10000
    throughputs = spec_trace.trace[order].data['THROUGHPUT']

    c1s, c2s = [], []
    for wl, wu in zip(wavebin_low * 10000, wavebin_up * 10000):
        wr = [wl, wu]
        try:
            c1, c2 = sld.compute_quadratic_ld_coeffs(wr, mode, wavelengths,
                                                     throughputs)
        except ValueError:
            c1, c2 = np.nan, np.nan
        c1s.append(c1)
        c2s.append(c2)
    c1s = np.array(c1s)
    c2s = np.array(c2s)

    return c1s, c2s


def run_stage4():
    return

# TODO: Add main to just run stage 4
if __name__ == "__main__":
    run_stage4()
