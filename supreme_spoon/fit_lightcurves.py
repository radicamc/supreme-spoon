#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 14:35 2022

@author: MCR

Juliet light curve fitting script
"""

from astropy.io import fits
import copy
import juliet
import matplotlib.backends.backend_pdf
import numpy as np
import pandas as pd

from supreme_spoon import stage4
from supreme_spoon import plotting, utils

# =============== User Input ===============
# Root file directory
root_dir = './'
# Name tag for output file directory.
output_tag = ''
# File containing lightcurves to fit.
infile = ''
# Orders to fit.
orders = [1, 2]
# Suffix to apply to fit output files.
fit_suffix = ''
# Integrations of ingress and egress.
baseline_ints = [50, -50]
# Type of occultation: 'transit' or 'eclipse'.
occultation_type = 'transit'
# If True, make summary plots.
do_plots = True
# Number of cores for multiprocessing.
ncores = 4
# Spectral resolution at which to fit lightcurves.
res = 'native'
# Planet identifier.
planet_letter = 'b'

# Fitting priors in juliet format.
params = ['P_p1', 't0_p1', 'p_p1', 'b_p1',
          'q1_SOSS', 'q2_SOSS', 'ecc_p1', 'omega_p1', 'a_p1',
          'mdilution_SOSS', 'mflux_SOSS', 'sigma_w_SOSS']
dists = ['fixed', 'fixed', 'uniform', 'fixed',
         'uniform', 'uniform', 'fixed', 'fixed', 'fixed',
         'fixed', 'fixed', 'loguniform']
hyperps = [3.42525650, 2459751.821681146, [0., 1], 0.748,
           [0., 1.], [0., 1.], 0.0, 90., 8.82,
           1.0, 0, [1e-1, 1e4]]

# Paths to files containing model limb-darkening coefficients.
ldcoef_file_o1 = None
ldcoef_file_o2 = None
# Path to file containing linear detrending parameters.
lm_file = None
# Key names for detrending parametrers.
lm_parameters = ['x']
# Path to file containing GP training parameters.
gp_file = None
# Key names for GP training parametrers.
gp_parameters = []
# ==========================================

if output_tag != '':
    output_tag = '_' + output_tag
# Create output directories and define output paths.
utils.verify_path(root_dir + 'pipeline_outputs_directory' + output_tag)
utils.verify_path(root_dir + 'pipeline_outputs_directory' + output_tag + '/Stage4')
outdir = root_dir + 'pipeline_outputs_directory' + output_tag + '/Stage4/'

# Tag for this particular fit.
if fit_suffix != '':
    fit_suffix = '_' + fit_suffix
# Add resolution info to the fit tag.
if res == 'native':
    fit_suffix += '_native'
    res_str = 'native resolution'
elif res == 'pixel':
    fit_suffix += '_pixel'
    res_str = 'pixel resolution'
else:
    fit_suffix += '_R{}'.format(res)
    res_str = 'R = {}'.format(res)

# Formatted parameter names for plotting.
formatted_names = {'P_p1': r'$P$', 't0_p1': r'$T_0$', 'p_p1': r'$R$_p/R_*$',
                   'b_p1': r'$b$', 'q1_SOSS': r'$q_1$', 'q2_SOSS': r'$q_2$',
                   'ecc_p1': r'$e$', 'omega_p1': r'$\Omega$',
                   'a_p1': r'$a/R_*$', 'sigma_w': r'$\sigma_w_SOSS$',
                   'theta0_SOSS': r'$\theta_0$', 'theta1_SOSS': r'$\theta_1$',
                   'theta2_SOSS': r'$\theta_2$',
                   'GP_sigma_SOSS': r'$GP_\sigma$', 'GP_rho_SOSS': r'$GP_rho$',
                   'rho': r'$\rho$'}

# === Get Detrending Quantities ===
# Get time axis
t = fits.getdata(infile, 9)
# Quantities against which to linearly detrend.
if lm_file is not None:
    lm_data = pd.read_csv(lm_file, comment='#')
    lm_quantities = np.zeros((len(t), len(lm_parameters)+1))
    lm_quantities[:, 0] = np.ones_like(t)
    for i, key in enumerate(lm_parameters):
        lm_param = lm_data[key]
        lm_quantities[:, i] = (lm_param - np.mean(lm_param)) / np.sqrt(np.var(lm_param))
# Quantities on which to train GP.
if gp_file is not None:
    gp_data = pd.read_csv(gp_file, comment='#')
    gp_quantities = np.zeros((len(t), len(gp_parameters)+1))
    gp_quantities[:, 0] = np.ones_like(t)
    for i, key in enumerate(gp_parameters):
        gp_param = gp_data[key]
        gp_quantities[:, i] = (gp_param - np.mean(gp_param)) / np.sqrt(np.var(gp_param))

# Format the baseline frames - either out-of-transit or in-eclipse.
baseline_ints = utils.format_out_frames(baseline_ints,
                                        occultation_type)

# === Fit Light Curves ===
# Start the light curve fitting.
results_dict = {}
for order in orders:
    first_time = True
    if do_plots is True:
        outpdf = matplotlib.backends.backend_pdf.PdfPages(outdir + 'lightcurve_fit_order{0}{1}.pdf'.format(order, fit_suffix))
    else:
        outpdf = None

    # === Set Up Priors and Fit Parameters ===
    print('\nFitting order {} at {}\n'.format(order, res_str))
    # Unpack wave, flux and error
    wave_low = fits.getdata(infile,  1 + 4*(order - 1))
    wave_up = fits.getdata(infile, 2 + 4*(order - 1))
    wave = np.nanmean(np.stack([wave_low, wave_up]), axis=0)
    flux = fits.getdata(infile, 3 + 4*(order - 1))
    err = fits.getdata(infile, 4 + 4*(order - 1))

    # Bin input spectra to desired resolution.
    if res == 'pixel':
        wave, wave_low, wave_up = wave[0], wave_low[0], wave_up[0]
    elif res == 'native':
        binned_vals = stage4.bin_at_pixel(flux, err, wave, npix=2)
        wave, wave_low, wave_up, flux, err = binned_vals
        wave, wave_low, wave_up = wave[0], wave_low[0], wave_up[0]
    else:
        binned_vals = stage4.bin_2d_spectra(wave, flux, err, res=res)
        wave, wave_low, wave_up, flux, err = binned_vals
        wave, wave_low, wave_up = wave[0], wave_low[0], wave_up[0]

    # For order 2, only fit wavelength bins between 0.6 and 0.85µm.
    if order == 2:
        ii = np.where((wave >= 0.6) & (wave <= 0.85))[0]
        flux, err = flux[:, ii], err[:, ii]
        wave, wave_low, wave_up = wave[ii], wave_low[ii], wave_up[ii]
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
    for param, dist, hyperp in zip(params, dists, hyperps):
        priors[param] = {}
        priors[param]['distribution'] = dist
        priors[param]['hyperparameters'] = hyperp
    # Interpolate LD coefficients from stellar models.
    if order == 1 and ldcoef_file_o1 is not None:
        prior_q1, prior_q2 = utils.read_ld_coefs(ldcoef_file_o1, wave_low,
                                                 wave_up)
    if order == 2 and ldcoef_file_o2 is not None:
        prior_q1, prior_q2 = utils.read_ld_coefs(ldcoef_file_o2, wave_low,
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
        # Update the LD prior for this bin if available.
        if ldcoef_file_o1 is not None or ldcoef_file_o2 is not None:
            if np.isfinite(prior_q1[wavebin]):
                prior_dict[thisbin]['q1_SOSS']['distribution'] = 'truncatednormal'
                prior_dict[thisbin]['q1_SOSS']['hyperparameters'] = [prior_q1[wavebin], 0.1, 0.0, 1.0]
            if np.isfinite(prior_q2[wavebin]):
                prior_dict[thisbin]['q2_SOSS']['distribution'] = 'truncatednormal'
                prior_dict[thisbin]['q2_SOSS']['hyperparameters'] = [prior_q2[wavebin], 0.1, 0.0, 1.0]

    # === Do the Fit ===
    # Fit each light curve
    fit_results = stage4.fit_lightcurves(data_dict, prior_dict, order=order,
                                         output_dir=outdir, nthreads=ncores,
                                         fit_suffix=fit_suffix)

    # === Summarize Fit Results ===
    # Loop over results for each wavebin, extract best-fitting parameters and
    # make summary plots if necessary.
    print('Summarizing fit results.')
    data = np.ones((nints, nbins)) * np.nan
    models = np.ones((nints, nbins)) * np.nan
    residuals = np.ones((nints, nbins)) * np.nan
    order_results = {'dppm': [], 'dppm_err': [], 'wave': wave,
                     'wave_err': np.mean([wave_low, wave_up], axis=0)}
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
            pp = fit_results[wavebin].posteriors['posterior_samples']['p_p1']
            md, up, lw = juliet.utils.get_quantiles(pp)
            order_results['dppm'].append((md**2)*1e6)
            err_low = (md**2 - lw**2)*1e6
            err_up = (up**2 - md**2)*1e6
            order_results['dppm_err'].append(np.mean([err_up, err_low]))

        # Make summary plots.
        if skip is False and do_plots is True:
            try:
                # Plot transit model and residuals.
                transit_model = fit_results[wavebin].lc.evaluate('SOSS')
                scatter = np.median(fit_results[wavebin].posteriors['posterior_samples']['sigma_w_SOSS'])
                nfit = len(np.where(dists != 'fixed')[0])
                t0 = np.median(fit_results.posteriors.posterior_samples['t0_p1'])
                plotting.do_lightcurve_plot(t=(t-t0)*24, data=norm_flux[:, i],
                                            model=transit_model,
                                            scatter=scatter,
                                            errors=norm_err[:, i],
                                            outpdf=outpdf, nfit=nfit,
                                            title='bin {0} | {1:.3f}µm'.format(i, wave[i]))
                # Corner plot for fit.
                fit_params, posterior_names = [], []
                for param, dist in zip(params, dists):
                    if dist != 'fixed':
                        fit_params.append(param)
                        if param in formatted_names.keys():
                            posterior_names.append(formatted_names[param])
                        else:
                            posterior_names.append(param)
                plotting.make_corner(fit_params, fit_results[wavebin],
                                     outpdf=outpdf,
                                     posterior_names=posterior_names)

                data[:, i] = norm_flux[:, i]
                models[:, i] = transit_model
                residuals[:, i] = norm_flux[:, i] - transit_model
            except:
                pass
    results_dict['order {}'.format(order)] = order_results
    # Plot 2D lightcurves.
    if do_plots is True:
        plotting.plot_2dlightcurves(wave, data, outpdf=outpdf,
                                    title='Normalized Lightcurves')
        plotting.plot_2dlightcurves(wave, models, outpdf=outpdf,
                                    title='Model Lightcurves')
        plotting.plot_2dlightcurves(wave, residuals, outpdf=outpdf,
                                    title='Residuals')
        outpdf.close()

# === Transmission Spectrum ===
# Save the transmission spectrum.
print('Writing transmission spectrum.')
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
infile_header = fits.getheader(infile, 0)
extract_type = infile_header['METHOD']
target = infile_header['TARGET'] + planet_letter
filename = target + '_NIRISS_SOSS_' + extract_type + '_tranmission_spectrum' \
           + fit_suffix + '.csv'
# Get fit metadata.
# Include fixed parameter values.
fit_metadata = '#\n# Fit Metadata\n'
for param, dist, hyper in zip(params, dists, hyperps):
    if dist == 'fixed' and param not in ['mdilution_SOSS', 'mflux_SOSS']:
        fit_metadata += '# {}: {}\n'.format(formatted_names[param], hyper)
# Append info on detrending via linear models or GPs.
if len(lm_parameters) != 0:
    fit_metadata += '# Linear Model: '
    for i, param in enumerate(lm_parameters):
        if i == 0:
            fit_metadata += param
        else:
            fit_metadata += ', {}'.format(param)
    fit_metadata += '\n'
if len(gp_parameters) != 0:
    fit_metadata += '# Gaussian Process: '
    for i, param in enumerate(gp_parameters):
        if i == 0:
            fit_metadata += param
        else:
            fit_metadata += ', {}'.format(param)
    fit_metadata += '\n'
fit_metadata += '#\n'

# Save spectrum.
stage4.save_transmission_spectrum(waves, wave_errors, depths, errors, orders,
                                  outdir, filename=filename, target=target,
                                  extraction_type=extract_type,
                                  resolution=res, fit_meta=fit_metadata)
print('Transmission spectrum saved to {}'.format(outdir+filename))

print('Done')
