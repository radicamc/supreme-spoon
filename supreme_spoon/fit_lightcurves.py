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
# TODO: Some toggle to turn off plotting
from astropy.io import fits
import copy
import juliet
import matplotlib.backends.backend_pdf
import numpy as np
import pandas as pd

from supreme_spoon import stage4
from supreme_spoon import plotting, utils

# =============== User Input ===============
root_dir = './'
output_tag = ''
infile = ''
orders = [1, 2]
fit_suffix = ''
baseline_ints = [50, -50]
occultation_type = 'transit'

# Fitting priors
params = ['P_p1', 't0_p1', 'p_p1', 'b_p1',
          'q1_SOSS', 'q2_SOSS', 'ecc_p1', 'omega_p1', 'a_p1',
          'mdilution_SOSS', 'mflux_SOSS', 'sigma_w_SOSS']
dists = ['fixed', 'fixed', 'uniform', 'fixed',
         'uniform', 'uniform', 'fixed', 'fixed', 'fixed',
         'fixed', 'fixed', 'loguniform']
hyperps = [3.42525650, 2459751.821681146, [0., 1], 0.748,
           [0., 1.], [0., 1.], 0.0, 90., 8.82,
           1.0, 0, [1e-1, 1e4]]

ldcoef_file_o1 = None
ldcoef_file_o2 = None
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
lm_quantities = np.zeros((len(t), 2))
lm_quantities[:, 0] = np.ones_like(t)
lm_quantities[:, 1] = (t - np.mean(t)) / np.sqrt(np.var(t))
# Quantities on which to train GP.
gp_quantities = np.zeros((len(t), 1))

# Format the baseline frames - either out-of-transit or in-eclipse.
baseline_ints = utils.format_out_frames(baseline_ints,
                                        occultation_type)

for order in orders:

    first_time = True
    outpdf = matplotlib.backends.backend_pdf.PdfPages(outdir + 'lightcurve_fit_order{0}{1}.pdf'.format(order, fit_suffix))

    print('\nFitting order {}\n'.format(order))
    # Unpack wave, flux and error
    wave_low = fits.getdata(infile,  1 + 4*(order - 1))[0]
    wave_up = fits.getdata(infile, 2 + 4*(order - 1))[0]
    wave = np.nanmean(np.stack([wave_low, wave_up]), axis=0)
    flux = fits.getdata(infile, 3 + 4*(order - 1))
    err = fits.getdata(infile, 4 + 4*(order - 1))
    nints, nbins = np.shape(flux)
    # Normalize flux and error by the baseline.
    baseline = np.median(flux[baseline_ints], axis=0)
    norm_flux = flux / baseline
    norm_err = err / baseline

    # Sort input arrays in order of increasing wavelength.
    ii = np.argsort(wave)
    wave_low, wave_up, wave = wave_low[ii], wave_up[ii], wave[ii]
    norm_flux, norm_err = norm_flux[:, ii], norm_err[:, ii]

    # Set up priors
    priors = {}
    for param, dist, hyperp in zip(params, dists, hyperps):
        priors[param] = {}
        priors[param]['distribution'] = dist
        priors[param]['hyperparameters'] = hyperp
    # Interpolate LD coefficients from stellar models.
    if order == 1 and ldcoef_file_o1 is not None:
        prior_q1, prior_q2 = utils.get_ld_coefs(ldcoef_file_o1,
                                                wave_low, wave_up)
    if order == 2 and ldcoef_file_o2 is not None:
        prior_q1, prior_q2 = utils.get_ld_coefs(ldcoef_file_o2,
                                                wave_low, wave_up)

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
                                         output_dir=outdir, nthreads=4,
                                         fit_suffix=fit_suffix)

    # Loop over results for each wavebin, and make summary plots.
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
            # If not skipped, append median and 1 sigma bounds.
            else:
                pp = fit_results[wavebin].posteriors['posterior_samples'][param]
                pm, pu, pl = juliet.utils.get_quantiles(pp)
                outdict[param + '_m'].append(pm)
                outdict[param + '_u'].append(pu)
                outdict[param + '_l'].append(pl)
        first_time = False

        # Make summary plots
        if skip is False:
            try:
                transit_model = fit_results[wavebin].lc.evaluate('SOSS')
                scatter = np.median(fit_results[wavebin].posteriors['posterior_samples']['sigma_w_SOSS'])
                nfit = len(np.where(dists != 'fixed')[0])
                t0 = np.median(fit_results.posteriors.posterior_samples['t0_p1'])
                plotting.do_lightcurve_plot(t=(t-t0)*24, data=norm_flux[:, i],
                                            model=transit_model,
                                            scatter=scatter,
                                            errors=norm_err[:, i],
                                            outpdf=outpdf,
                                            title='bin {0} | {1:.3f}µm'.format(i, wave[i]),
                                            nfit=nfit)

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

    plotting.plot_2dlightcurves(wave, data, outpdf=outpdf,
                                title='Normalized Lightcurves')
    plotting.plot_2dlightcurves(wave, models, outpdf=outpdf,
                                title='Model Lightcurves')
    plotting.plot_2dlightcurves(wave, residuals, outpdf=outpdf,
                                title='Residuals')
    outdf = pd.DataFrame(data=outdict)
    outdf.to_csv(outdir + 'speclightcurve_results_order{0}{1}.csv'.format(order, fit_suffix),
                 index=False)
    outpdf.close()

print('Done')
