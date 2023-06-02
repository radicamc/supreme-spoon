#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 14:35 2022

@author: MCR

Juliet light curve fitting script.
"""

from astropy.io import fits
import copy
from datetime import datetime
import glob
import juliet
import matplotlib.backends.backend_pdf
import numpy as np
import os
import pandas as pd
import shutil
import sys

from supreme_spoon import stage4
from supreme_spoon import plotting, utils
from supreme_spoon.utils import fancyprint, verify_path

# Read config file.
try:
    config_file = sys.argv[1]
except IndexError:
    msg = 'Config file must be provided'
    raise FileNotFoundError(msg)
config = utils.parse_config(config_file)

if config['output_tag'] != '':
    config['output_tag'] = '_' + config['output_tag']
# Create output directories and define output paths.
utils.verify_path('pipeline_outputs_directory' + config['output_tag'])
utils.verify_path('pipeline_outputs_directory' + config['output_tag'] + '/Stage4')
outdir = 'pipeline_outputs_directory' + config['output_tag'] + '/Stage4/'

# Get all files in output directory for checks.
all_files = glob.glob(outdir + '*')

# Tag for this particular fit.
if config['fit_suffix'] != '':
    fit_suffix = '_' + config['fit_suffix']
else:
    fit_suffix = config['fit_suffix']
# Add resolution/binning info to the fit tag.
if config['res'] is not None:
    if config['res'] == 'pixel':
        fit_suffix += '_pixel'
        res_str = 'pixel resolution'
    else:
        fit_suffix += '_R{}'.format(config['res'])
        res_str = 'R = {}'.format(config['res'])
elif config['npix'] is not None:
    fit_suffix += '_{}pix'.format(config['npix'])
    res_str = 'npix = {}'.format(config['npix'])
else:
    msg = 'Number of columns to bin or spectral resolution must be provided.'
    raise ValueError(msg)

# Save a copy of the config file.
root_dir = 'pipeline_outputs_directory' + config['output_tag'] + '/config_files'
verify_path(root_dir)
i = 0
copy_config = root_dir + '/' + config_file
root = copy_config.split('.yaml')[0]
copy_config = root + '{}.yaml'.format(fit_suffix)
while os.path.exists(copy_config):
    i += 1
    copy_config = root_dir + '/' + config_file
    root = copy_config.split('.yaml')[0]
    copy_config = root + '{0}_{1}.yaml'.format(fit_suffix, i)
shutil.copy(config_file, copy_config)
# Append time at which it was run.
f = open(copy_config, 'a')
time = datetime.utcnow().isoformat(sep=' ', timespec='minutes')
f.write('\nRun at {}.'.format(time))
f.close()

# Formatted parameter names for plotting.
formatted_names = {'P_p1': r'$P$', 't0_p1': r'$T_0$', 'p_p1': r'$R_p/R_*$',
                   'b_p1': r'$b$', 'q1_SOSS': r'$q_1$', 'q2_SOSS': r'$q_2$',
                   'ecc_p1': r'$e$', 'omega_p1': r'$\Omega$',
                   'a_p1': r'$a/R_*$', 'sigma_w_SOSS': r'$\sigma_w$',
                   'theta0_SOSS': r'$\theta_0$', 'theta1_SOSS': r'$\theta_1$',
                   'theta2_SOSS': r'$\theta_2$',
                   'GP_sigma_SOSS': r'$GP \sigma$',
                   'GP_rho_SOSS': r'$GP \rho$', 'rho': r'$\rho$',
                   't_secondary_p1': r'$T_{sec}$', 'fp_p1': r'$F_p/F_*$'}

# === Get Detrending Quantities ===
# Get time axis
t = fits.getdata(config['infile'], 9)
# Quantities against which to linearly detrend.
if config['lm_file'] is not None:
    lm_data = pd.read_csv(config['lm_file'], comment='#')
    lm_quantities = np.zeros((len(t), len(config['lm_parameters'])+1))
    lm_quantities[:, 0] = np.ones_like(t)
    for i, key in enumerate(config['lm_parameters']):
        lm_param = lm_data[key]
        lm_quantities[:, i] = (lm_param - np.mean(lm_param)) / np.sqrt(np.var(lm_param))
# Eclipses must fit for a baseline, which is done via the linear detrending.
# So add this term to the fits if not already included.
if config['lm_file'] is None and config['occultation_type'] == 'eclipse':
    lm_quantities = np.zeros((len(t), 1))
    lm_quantities[:, 0] = np.ones_like(t)
    config['params'].append('theta0_SOSS')
    config['dists'].append('uniform')
    config['hyperps'].append([-10, 10])

# Quantity on which to train GP.
if config['gp_file'] is not None:
    gp_data = pd.read_csv(config['gp_file'], comment='#')
    gp_quantities = gp_data[config['gp_parameter']].values

# Format the baseline frames - either out-of-transit or in-eclipse.
baseline_ints = utils.format_out_frames(config['baseline_ints'],
                                        config['occultation_type'])

# === Fit Light Curves ===
# Start the light curve fitting.
results_dict = {}
for order in config['orders']:
    first_time = True
    expected_file = outdir + 'lightcurve_fit_order{0}{1}.pdf'.format(order, fit_suffix)
    if config['do_plots'] is True and expected_file not in all_files:
        outpdf = matplotlib.backends.backend_pdf.PdfPages(expected_file)
        doplot = True
    else:
        doplot = False
        outpdf = None

    # === Set Up Priors and Fit Parameters ===
    fancyprint('Fitting order {} at {}.'.format(order, res_str))
    # Unpack wave, flux and error, cutting reference pixel columns.
    wave_low = fits.getdata(config['infile'],  1 + 4*(order - 1))[:, 5:-5]
    wave_up = fits.getdata(config['infile'], 2 + 4*(order - 1))[:, 5:-5]
    wave = np.nanmean(np.stack([wave_low, wave_up]), axis=0)
    flux = fits.getdata(config['infile'], 3 + 4*(order - 1))[:, 5:-5]
    err = fits.getdata(config['infile'], 4 + 4*(order - 1))[:, 5:-5]

    # For order 2, only fit wavelength bins between 0.6 and 0.85µm.
    if order == 2:
        ii = np.where((wave[0] >= 0.6) & (wave[0] <= 0.85))[0]
        flux, err = flux[:, ii], err[:, ii]
        wave, wave_low, wave_up = wave[:, ii], wave_low[:, ii], wave_up[:, ii]

    # Bin input spectra to desired resolution.
    if config['res'] is not None:
        if config['res'] == 'pixel':
            wave, wave_low, wave_up = wave[0], wave_low[0], wave_up[0]
        else:
            binned_vals = stage4.bin_at_resolution(wave_low[0], wave_up[0],
                                                   flux.T, err.T,
                                                   res=config['res'])
            wave, wave_err, flux, err = binned_vals
            flux, err = flux.T, err.T
            wave_low, wave_up = wave - wave_err, wave + wave_err
    else:
        binned_vals = stage4.bin_at_pixel(flux, err, wave, npix=config['npix'])
        wave, wave_low, wave_up, flux, err = binned_vals
        wave, wave_low, wave_up = wave[0], wave_low[0], wave_up[0]
    nints, nbins = np.shape(flux)

    # Sort input arrays in order of increasing wavelength.
    ii = np.argsort(wave)
    wave_low, wave_up, wave = wave_low[ii], wave_up[ii], wave[ii]
    flux, err = flux[:, ii], err[:, ii]

    # Normalize flux and error by the baseline.
    baseline = np.median(flux[baseline_ints], axis=0)
    norm_flux = flux / baseline
    norm_err = err / baseline

    # Set up priors
    priors = {}
    for param, dist, hyperp in zip(config['params'], config['dists'], config['hyperps']):
        priors[param] = {}
        priors[param]['distribution'] = dist
        priors[param]['hyperparameters'] = hyperp
    # For transit fits, interpolate LD coefficients from stellar models.
    if config['occultation_type'] == 'transit':
        if order == 1 and config['ldcoef_file_o1'] is not None:
            q1, q2 = stage4.read_ld_coefs(config['ldcoef_file_o1'], wave_low,
                                          wave_up)
        if order == 2 and config['ldcoef_file_o2'] is not None:
            q1, q2 = stage4.read_ld_coefs(config['ldcoef_file_o2'], wave_low,
                                          wave_up)

    # Pack fitting arrays and priors into dictionaries.
    data_dict, prior_dict = {}, {}
    for wavebin in range(nbins):
        # Data dictionaries, including linear model and GP regressors.
        thisbin = 'wavebin' + str(wavebin)
        bin_dict = {'times': t,
                    'flux': norm_flux[:, wavebin],
                    'error': np.zeros_like(norm_err[:, wavebin])}
        # If linear models are to be included.
        if 'theta0_SOSS' in priors.keys():
            bin_dict['lm_parameters'] = lm_quantities
        # If GPs are to be inclided.
        if 'GP_sigma_SOSS' in priors.keys():
            bin_dict['GP_parameters'] = gp_quantities
        data_dict[thisbin] = bin_dict

        # Prior dictionaries.
        prior_dict[thisbin] = copy.deepcopy(priors)
        # For transit only; update the LD prior for this bin if available.
        if config['occultation_type'] == 'transit':
            # Set prior width to 0.2 around the model value - based on
            # findings of Patel & Espinoza 2022.
            if config['ldcoef_file_o1'] is not None or config['ldcoef_file_o2'] is not None:
                if np.isfinite(q1[wavebin]):
                    prior_dict[thisbin]['q1_SOSS']['distribution'] = 'truncatednormal'
                    prior_dict[thisbin]['q1_SOSS']['hyperparameters'] = [q1[wavebin], 0.2, 0.0, 1.0]
                if np.isfinite(q2[wavebin]):
                    prior_dict[thisbin]['q2_SOSS']['distribution'] = 'truncatednormal'
                    prior_dict[thisbin]['q2_SOSS']['hyperparameters'] = [q2[wavebin], 0.2, 0.0, 1.0]

    # === Do the Fit ===
    # Fit each light curve
    fit_results = stage4.fit_lightcurves(data_dict, prior_dict, order=order,
                                         output_dir=outdir,
                                         nthreads=config['ncores'],
                                         fit_suffix=fit_suffix)

    # === Summarize Fit Results ===
    # Loop over results for each wavebin, extract best-fitting parameters and
    # make summary plots if necessary.
    fancyprint('Summarizing fit results.')
    data = np.ones((nints, nbins)) * np.nan
    models = np.ones((nints, nbins)) * np.nan
    residuals = np.ones((nints, nbins)) * np.nan
    order_results = {'dppm': [], 'dppm_err': [], 'wave': wave,
                     'wave_err': np.mean([wave - wave_low, wave_up - wave],
                                         axis=0)}
    for i, wavebin in enumerate(fit_results.keys()):
        # Make note if something went wrong with this bin.
        skip = False
        if fit_results[wavebin] is None:
            skip = True

        # Pack best fit Rp/R* into a dictionary.
        # Append NaNs if the bin was skipped.
        if skip is True:
            order_results['dppm'].append(np.nan)
            order_results['dppm_err'].append(np.nan)
        # If not skipped, append median and 1-sigma bounds.
        else:
            post_samples = fit_results[wavebin].posteriors['posterior_samples']
            if config['occultation_type'] == 'transit':
                md, up, lw = juliet.utils.get_quantiles(post_samples['p_p1'])
                order_results['dppm'].append((md**2)*1e6)
                err_low = (md**2 - lw**2)*1e6
                err_up = (up**2 - md**2)*1e6
            else:
                md, up, lw = juliet.utils.get_quantiles(post_samples['fp_p1'])
                order_results['dppm'].append(md*1e6)
                err_low = (md - lw)*1e6
                err_up = (up - md)*1e6
            order_results['dppm_err'].append(np.mean([err_up, err_low]))

        # Make summary plots.
        if skip is False and doplot is True:
            try:
                # Plot transit model and residuals.
                if config['gp_file'] is not None:
                    # Hack to get around weird bug where ray fits with GPs end
                    # up being read only.
                    outdir_i = outdir + 'speclightcurve{2}/order{0}_{1}'.format(order, wavebin, fit_suffix)
                    dataset = juliet.load(priors=prior_dict[wavebin],
                                          t_lc={'SOSS': data_dict[wavebin]['times']},
                                          y_lc={'SOSS': data_dict[wavebin]['flux']},
                                          yerr_lc={'SOSS': data_dict[wavebin]['error']},
                                          GP_regressors_lc={'SOSS': data_dict[wavebin]['GP_parameters']},
                                          out_folder=outdir_i)
                    results = dataset.fit(sampler='dynesty')
                    transit_model, comp = results.lc.evaluate('SOSS',
                                                              GPregressors=t,
                                                              return_components=True)
                else:
                    transit_model, comp = fit_results[wavebin].lc.evaluate('SOSS', return_components=True)
                scatter = np.median(fit_results[wavebin].posteriors['posterior_samples']['sigma_w_SOSS'])
                nfit = len(np.where(config['dists'] != 'fixed')[0])
                t0_loc = np.where(np.array(config['params']) == 't0_p1')[0][0]
                if config['dists'][t0_loc] == 'fixed':
                    t0 = config['hyperps'][t0_loc]
                else:
                    t0 = np.median(fit_results[wavebin].posteriors['posterior_samples']['t0_p1'])

                # Get systematics and transit models.
                systematics = None
                if config['gp_file'] is not None or config['lm_file'] is not None:
                    if config['gp_file'] is not None:
                        gp_model = transit_model - comp['transit'] - comp['lm']
                    else:
                        gp_model = np.zeros_like(t)
                    systematics = gp_model + comp['lm']
                plotting.make_lightcurve_plot(t=(t - t0)*24,
                                              data=norm_flux[:, i],
                                              model=transit_model,
                                              scatter=scatter,
                                              errors=norm_err[:, i],
                                              outpdf=outpdf, nfit=nfit,
                                              title='bin {0} | {1:.3f}µm'.format(i, wave[i]),
                                              systematics=systematics,
                                              rasterized=True)
                # Corner plot for fit.
                if config['include_corner'] is True:
                    fit_params, posterior_names = [], []
                    for param, dist in zip(config['params'], config['dists']):
                        if dist != 'fixed':
                            fit_params.append(param)
                            if param in formatted_names.keys():
                                posterior_names.append(formatted_names[param])
                            else:
                                posterior_names.append(param)
                    plotting.make_corner_plot(fit_params, fit_results[wavebin],
                                              posterior_names=posterior_names,
                                              outpdf=outpdf)

                data[:, i] = norm_flux[:, i]
                models[:, i] = transit_model
                residuals[:, i] = norm_flux[:, i] - transit_model
            except:
                pass
    results_dict['order {}'.format(order)] = order_results
    # Plot 2D lightcurves.
    if doplot is True:
        plotting.make_2d_lightcurve_plot(wave, data, outpdf=outpdf,
                                         title='Normalized Lightcurves')
        plotting.make_2d_lightcurve_plot(wave, models, outpdf=outpdf,
                                         title='Model Lightcurves')
        plotting.make_2d_lightcurve_plot(wave, residuals, outpdf=outpdf,
                                         title='Residuals')
        outpdf.close()

# === Transmission Spectrum ===
# Save the transmission spectrum.
fancyprint('Writing transmission spectrum.')
for order in ['1', '2']:
    if 'order '+order not in results_dict.keys():
        order_results = {'dppm': [], 'dppm_err': [], 'wave': [],
                         'wave_err': []}
        results_dict['order '+order] = order_results

# Concatenate transit depths, wavelengths, and associated errors from both
# orders.
depths = np.concatenate([results_dict['order 2']['dppm'],
                         results_dict['order 1']['dppm']])
errors = np.concatenate([results_dict['order 2']['dppm_err'],
                         results_dict['order 1']['dppm_err']])
waves = np.concatenate([results_dict['order 2']['wave'],
                        results_dict['order 1']['wave']])
wave_errors = np.concatenate([results_dict['order 2']['wave_err'],
                              results_dict['order 1']['wave_err']])
orders = np.concatenate([2*np.ones_like(results_dict['order 2']['dppm']),
                         np.ones_like(results_dict['order 1']['dppm'])]).astype(int)

# Get target/reduction metadata.
infile_header = fits.getheader(config['infile'], 0)
extract_type = infile_header['METHOD']
target = infile_header['TARGET'] + config['planet_letter']
if config['occultation_type'] == 'transit':
    spec_type = 'transmission'
else:
    spec_type = 'emission'
filename = target + '_NIRISS_SOSS_' + spec_type + '_spectrum' + fit_suffix + '.csv'
# Get fit metadata.
# Include fixed parameter values.
fit_metadata = '#\n# Fit Metadata\n'
for param, dist, hyper in zip(config['params'], config['dists'], config['hyperps']):
    if dist == 'fixed' and param not in ['mdilution_SOSS', 'mflux_SOSS']:
        fit_metadata += '# {}: {}\n'.format(formatted_names[param], hyper)
# Append info on detrending via linear models or GPs.
if len(config['lm_parameters']) != 0:
    fit_metadata += '# Linear Model: '
    for i, param in enumerate(config['lm_parameters']):
        if i == 0:
            fit_metadata += param
        else:
            fit_metadata += ', {}'.format(param)
    fit_metadata += '\n'
if config['gp_parameter'] != '':
    fit_metadata += '# Gaussian Process: '
    fit_metadata += config['gp_parameter']
    fit_metadata += '\n'
fit_metadata += '#\n'

# Save spectrum.
stage4.save_transmission_spectrum(waves, wave_errors, depths, errors, orders,
                                  outdir, filename=filename, target=target,
                                  extraction_type=extract_type,
                                  resolution=config['res'],
                                  fit_meta=fit_metadata,
                                  occultation_type=config['occultation_type'])
fancyprint('{0} spectrum saved to {1}'.format(spec_type, outdir+filename))

fancyprint('Done')
