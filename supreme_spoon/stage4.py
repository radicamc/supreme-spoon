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
import juliet
import ray
from tqdm import tqdm
import numpy as np

from jwst import datamodels
from jwst.pipeline import calwebb_spec2


def fit_lightcurves(data_dict, prior_dict, order, output_dir, fit_suffix,
                    nthreads=4):

    # Initialize results dictionary and keynames.
    results = dict.fromkeys(data_dict.keys(), [])
    keynames = list(data_dict.keys())

    # Format output directory
    if output_dir[-1] != '/':
        output_dir += '/'

    # Initialize ray with specified number of threads.
    ray.shutdown()
    ray.init(num_cpus=nthreads)

    # Set juliet fits as remotes to run parallel with ray.
    all_fits = []
    num_bins = np.arange(len(keynames))+1
    for i, keyname in enumerate(keynames):
        outdir = output_dir + 'speclightcurve{2}/order{0}_{1}'.format(order, keyname, fit_suffix)
        all_fits.append(fit_data.remote(data_dict[keyname],
                                        prior_dict[keyname],
                                        output_dir=outdir,
                                        bin_no=num_bins[i],
                                        num_bins=len(num_bins)))
    # Run the fits.
    ray_results = ray.get(all_fits)

    # Reorder the results based on the key name.
    for i in range(len(keynames)):
        keyname = keynames[i]
        results[keyname] = ray_results[i]

    return results


@ray.remote
def fit_data(data_dictionary, priors, output_dir, bin_no, num_bins):

    print('Fitting bin {} / {}'.format(bin_no, num_bins))

    # Get key names.
    all_keys = list(data_dictionary.keys())

    # Unpack fitting arrays
    t = {'SOSS': data_dictionary['times']}
    flux = {'SOSS': data_dictionary['flux']}
    flux_err = {'SOSS': data_dictionary['error']}

    # Initialize GP and linear model regressors.
    gp_regressors = None
    linear_regressors = None
    if 'GP_parameters' in all_keys:
        gp_regressors = {'SOSS': data_dictionary['GP_parameters']}
    if 'lm_parameters' in all_keys:
        linear_regressors = {'SOSS': data_dictionary['lm_parameters']}

    fit_results = run_juliet(priors, t, flux, flux_err, output_dir,
                             gp_regressors, linear_regressors)

    return fit_results


def run_juliet(priors, t_lc, y_lc, yerr_lc, out_folder,
               gp_regressors_lc, linear_regressors_lc):

    if np.isfinite(y_lc['SOSS']).all():
        # Load in all priors and data to be fit.
        dataset = juliet.load(priors=priors, t_lc=t_lc, y_lc=y_lc,
                              yerr_lc=yerr_lc, out_folder=out_folder,
                              GP_regressors_lc=gp_regressors_lc,
                              linear_regressors_lc=linear_regressors_lc)
        kwargs = {'maxiter': 25000, 'print_progress': False}

        # Run the fit.
        try:
            res = dataset.fit(sampler='dynesty', **kwargs)
        except KeyboardInterrupt as err:
            raise err
        except:
            print('Exception encountered.')
            print('Skipping bin.')
            res = None
    else:
        print('NaN bin encountered.')
        print('Skipping bin.')
        res = None

    return res


def bin_at_resolution(wavelengths, depths, depth_error, R, method='sum'):
    """Function that bins input wavelengths and transit depths (or any other
    observable, like flux) to a given resolution `R`. Useful for binning
    transit depths down to a target resolution on a transit spectrum.
    Adapted from Néstor Espinoza

    Parameters
    ----------
    wavelengths : array-like[float]
        Wavelength values.
    depths : array-like[float]
        Depth values at each wavelength.
    depth_error : array-like[float]
        Errors corresponding to each depth measurement.
    R : int
        Target resolution at which to bin.
    method : str
        Method to bin depths.

    Returns
    -------
    wout : np.ndarray[float]
        Wavelength of the given bin at resolution R.
    werrout : list[float]
        Width of the wavelength bin.
    dout : np.ndarray[float]
        Binned depth.
    derrout : np.ndarray[float]
        Error on binned depth.
    """

    # Prepare output arrays:
    wout, dout, derrout = np.array([]), np.array([]), np.array([])

    # Sort wavelengths from lowest to highest:
    idx = np.argsort(wavelengths)
    ww = wavelengths[idx]
    dd = depths[idx]
    de = depth_error[idx]

    oncall = False
    # Loop over all wavelengths:
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
            current_r = np.mean(current_wavs) / np.abs(current_wavs[0] - current_wavs[-1])

            # If the current set of wavs/depths is below or at the target
            # resolution, stop and move to next bin:
            if current_r <= R:
                wout = np.append(wout, np.nanmean(current_wavs))
                if method == 'sum':
                    dout = np.append(dout, np.nansum(current_depths))
                    derrout = np.append(derrout, np.sqrt(np.nansum(current_errors**2)))
                elif method == 'average':
                    ii = np.where(~np.isfinite(current_depths) | ~np.isfinite(current_errors))
                    current_errors = np.delete(current_errors, ii)
                    current_depths = np.delete(current_depths, ii)
                    cd = np.average(current_depths, weights=1/current_errors**2)
                    dout = np.append(dout, cd)
                    derrout = np.append(derrout, np.sqrt(np.nansum(current_errors**2))/len(current_errors))
                else:
                    raise ValueError('Unidentified method {}.'.format(method))
                oncall = False

    # Calculate the wavelength limits of each bin.
    lw = np.concatenate([wout[:, None], np.roll(wout, 1)[:, None]], axis=1)
    up = np.concatenate([wout[:, None], np.roll(wout, -1)[:, None]], axis=1)

    uperr = (np.mean(up, axis=1) - wout)[:-1]
    uperr = np.insert(uperr, -1, uperr[-1])
    lwerr = (wout - np.mean(lw, axis=1))[1:]
    lwerr = np.insert(lwerr, 0, lwerr[0])
    werrout = [lwerr, uperr]

    return wout, werrout, dout, derrout


def bin_2d_spectra(wave2d, flux2d, err2d, R=150):
    """Utility to loop over bin_at_resolution for 2D dataframes (e.g., 2D
    spectroscopic lightcurves).

    Parameters
    ----------
    wave2d : array-like[float]
        2D array of wavelengths.
    flux2d : array-like[float]
        Flux values at to each wavelength.
    err2d : array-like[float]
        Errors on the correspondng fluxes.
    R : int
        Resolution to which to bin.

    Returns
    -------
    wc_bin : np.ndarray[float]
        Central wavelength of the bin.
    wl_bin : np.ndarray[float]
        Lower wavelength limit of the bin.
    wu_bin : np.ndarray[float]
        Upper wavelength limit of the bin.
    f_bin : np.ndarray[float]
        Binned flux.
    e_bin : np.ndarray[float]
        Binned error.
    """

    nints, nwave = np.shape(wave2d)

    # Simply loop over each integration and bin the flux values down to the
    # desired resolution.
    for i in tqdm(range(nints)):
        if i == 0:
            bin_res = bin_at_resolution(wave2d[i], flux2d[i], err2d[i], R=R)
            wc_bin, we_bin, f_bin, e_bin = bin_res
            wl_bin, wu_bin = we_bin
        elif i == 1:
            bin_res = bin_at_resolution(wave2d[i], flux2d[i], err2d[i], R=R)
            wc_bin_i, we_bin_i, f_bin_i, e_bin_i = bin_res
            wl_bin_i, wu_bin_i = we_bin_i
            wc_bin = np.stack([wc_bin, wc_bin_i])
            wl_bin = np.stack([wl_bin, wl_bin_i])
            wu_bin = np.stack([wu_bin, wu_bin_i])
            f_bin = np.stack([f_bin, f_bin_i])
            e_bin = np.stack([e_bin, e_bin_i])
        else:
            bin_res = bin_at_resolution(wave2d[i], flux2d[i], err2d[i], R=R)
            wc_bin_i, we_bin_i, f_bin_i, e_bin_i = bin_res
            wl_bin_i, wu_bin_i = we_bin_i
            wc_bin = np.concatenate([wc_bin, wc_bin_i[None, :]], axis=0)
            wl_bin = np.concatenate([wl_bin, wl_bin_i[None, :]], axis=0)
            wu_bin = np.concatenate([wu_bin, wu_bin_i[None, :]], axis=0)
            f_bin = np.concatenate([f_bin, f_bin_i[None, :]], axis=0)
            e_bin = np.concatenate([e_bin, e_bin_i[None, :]], axis=0)

    return wc_bin, wl_bin, wu_bin, f_bin, e_bin


def save_transmission_spectrum(wave, wave_err, dppm, dppm_err, order, outdir,
                               filename, target, extraction_type):
    """Write a transmission spectrum to file.

    Parameters
    ----------
    wave : array-like[float]
        Wavelength values.
    wave_err : array-like[float]
        Bin half-widths for each wavelength bin.
    dppm : array-like[float]
        Transit depth in each bin.
    dppm_err : array-like[float]
        Error on the transit depth in each bin.
    order : array-like[int]
        SOSS order corresponding to each bin.
    outdir : str
        Firectory to whch to save outputs.
    filename : str
        Name of the file to which to save spectra.
    target : str
        Target name.
    extraction_type : str
        Type of extraction: either box or atoca.
    """

    # Pack the quantities into a dictionary.
    dd = {'wave': wave,
          'wave_err': wave_err,
          'dppm': dppm,
          'dppm_err': dppm_err,
          'order': order}
    # Save the dictionary as a csv.
    df = pd.DataFrame(data=dd)
    if os.path.exists(outdir + filename):
        os.remove(outdir + filename)

    # Re-open the csv and append some critical info the header.
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
