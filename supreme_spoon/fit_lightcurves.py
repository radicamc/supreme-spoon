#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 14:35 2022

@author: MCR

Juliet light curve fitting script
"""
# TODO: funtions to fit white light curve
# TODO: procedural function run_stage4 to fit wlc then spec lcs
# TODO: fit_lightcurves.py as wrapper around run_Stage4 like run_DMS is for 1-3
# TODO: incorporate saving transmission spectrum
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
if isinstance(res, str):
    fit_suffix += '_native'
else:
    fit_suffix += 'R{}'.format(res)

formatted_names = {'P_p1': r'$P$', 't0_p1': r'$T_0$', 'p_p1': r'R$_p$/R$_*$',
                   'b_p1': r'$b$', 'q1_SOSS': r'$q_1$', 'q2_SOSS': r'$q_2$',
                   'ecc_p1': r'$e$', 'omega_p1': r'$\Omega$',
                   'sigma_w': r'$\sigma_w_SOSS$',
                   'theta0_SOSS': r'$\theta_0$', 'theta1_SOSS': r'$\theta_1$',
                   'theta2_SOSS': r'$\theta_2$',
                   'GP_sigma_SOSS': r'$GP_\sigma$', 'GP_rho_SOSS': r'$GP_rho$',
                   'rho': r'$\rho$'}

# Get time axis
t = fits.getdata(infile, 9)
# Quantities against which to linearly detrend.
if lm_file is not None:
    lm_data = pd.read_csv(lm_file, comments='#')
    lm_quantities = np.zeros((len(t), len(lm_parameters)+1))
    lm_quantities[:, 0] = np.ones_like(t)
    for i, key in enumerate(lm_parameters):
        lm_param = lm_data[key]
        lm_quantities[:, i] = (lm_param - np.mean(lm_param)) / np.sqrt(np.var(lm_param))
# Quantities on which to train GP.
gp_quantities = np.zeros((len(t), 1))
if gp_file is not None:
    gp_data = pd.read_csv(gp_file, comments='#')
    gp_quantities = np.zeros((len(t), len(gp_parameters)+1))
    gp_quantities[:, 0] = np.ones_like(t)
    for i, key in enumerate(gp_parameters):
        gp_param = gp_data[key]
        gp_quantities[:, i] = (gp_param - np.mean(gp_param)) / np.sqrt(np.var(gp_param))

# Format the baseline frames - either out-of-transit or in-eclipse.
baseline_ints = utils.format_out_frames(baseline_ints,
                                        occultation_type)

# Start the light curve fitting.
for order in orders:
    first_time = True
    if do_plots is True:
        outpdf = matplotlib.backends.backend_pdf.PdfPages(outdir + 'lightcurve_fit_order{0}{1}.pdf'.format(order, fit_suffix))
    else:
        outpdf = None

    print('\nFitting order {}\n'.format(order))
    # Unpack wave, flux and error
    wave_low = fits.getdata(infile,  1 + 4*(order - 1))
    wave_up = fits.getdata(infile, 2 + 4*(order - 1))
    wave = np.nanmean(np.stack([wave_low, wave_up]), axis=0)
    flux = fits.getdata(infile, 3 + 4*(order - 1))
    err = fits.getdata(infile, 4 + 4*(order - 1))

    # Bin input spectra to desired resolution.
    if res == 'native':
        binned_vals = stage4.bin_at_pixel(flux, err, wave, npix=2)
        wave, wave_low, wave_up, flux, err = binned_vals
        wave, wave_low, wave_up = wave[0], wave_low[0], wave_up[0]
    else:
        binned_vals = stage4.bin_2d_spectra(wave, flux, err, R=res)
        wave, wave_low, wave_up, flux, err = binned_vals
        wave, wave_low, wave_up = wave[0], wave_low[0], wave_up[0]

    # For order 2, only fit wavelength bins between 0.6 and 0.85µm.
    if order == 2:
        ii = np.where((wave >= 0.6) & (wave <= 0.85))
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

    # Fit each light curve
    fit_results = stage4.fit_lightcurves(data_dict, prior_dict, order=order,
                                         output_dir=outdir, nthreads=ncores,
                                         fit_suffix=fit_suffix)

    # Loop over results for each wavebin, extract best-fitting parameters and
    # make summary plots if necessary.
    print('Making summary plots.')
    data = np.ones((nints, nbins)) * np.nan
    models = np.ones((nints, nbins)) * np.nan
    residuals = np.ones((nints, nbins)) * np.nan
    outdict = {}
    for i, wavebin in enumerate(fit_results.keys()):
        # Make note if something went wrong with this bin.
        skip = False
        if fit_results[wavebin] is None:
            skip = True

        # Pack best fit params into a dictionary
        for param, dist in zip(params, dists):
            if dist == 'fixed':
                continue
            if first_time is True:
                outdict[param + '_m'] = []
                outdict[param + '_u'] = []
                outdict[param + '_l'] = []
            # Append NaNs if the bin was skipped.
            if skip is True:
                outdict[param + '_m'].append(np.nan)
                outdict[param + '_u'].append(np.nan)
                outdict[param + '_l'].append(np.nan)
            # If not skipped, append median and 1-sigma bounds.
            else:
                pp = fit_results[wavebin].posteriors['posterior_samples'][param]
                pm, pu, pl = juliet.utils.get_quantiles(pp)
                outdict[param + '_m'].append(pm)
                outdict[param + '_u'].append(pu)
                outdict[param + '_l'].append(pl)
        first_time = False

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

    # Save fit results to csv file.
    outdf = pd.DataFrame(data=outdict)
    outdf.to_csv(outdir + 'speclightcurve_results_order{0}{1}.csv'.format(order, fit_suffix),
                 index=False)

    # Plot 2D lightcurves.
    if do_plots is True:
        plotting.plot_2dlightcurves(wave, data, outpdf=outpdf,
                                    title='Normalized Lightcurves')
        plotting.plot_2dlightcurves(wave, models, outpdf=outpdf,
                                    title='Model Lightcurves')
        plotting.plot_2dlightcurves(wave, residuals, outpdf=outpdf,
                                    title='Residuals')
        outpdf.close()


# SAVE TRANSMISSION SPECTRUM HERE

print('Done')
