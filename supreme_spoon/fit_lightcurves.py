#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 14:35 2022

@author: MCR

Juliet light curve fitting script
"""

from astropy.io import fits
import juliet
import matplotlib.backends.backend_pdf
import numpy as np
import pandas as pd

from supreme_spoon import plotting, utils

# =============== User Input ===============
outdir = 'pipeline_outputs_directory/Stage4/'
orders = [1, 2]
suffix = 'box_R300'
out_frames = [90, -40]

# Fitting priors
params = ['P_p1', 't0_p1', 'p_p1', 'b_p1',
          'q1_SOSS', 'q2_SOSS', 'ecc_p1', 'omega_p1', 'a_p1',
          'mdilution_SOSS', 'mflux_SOSS', 'sigma_w_SOSS', 'theta0_SOSS']
dists = ['fixed', 'fixed', 'uniform', 'fixed',
         'uniform', 'uniform', 'fixed', 'fixed', 'fixed',
         'fixed', 'fixed', 'loguniform', 'normal']
hyperps = [3.42525650, 2459751.821681146, [0., 1], 0.748,
           [0., 1.], [0., 1.], 0.0, 90., 8.82,
           1.0, 0, [1e-1, 1e4], (9.1164330491e-05, 1.87579835e-05)]

nestorprior_o1 = None
nestorprior_o2 = None
# ==========================================

if suffix != '':
    suffix = '_' + suffix

formatted_names = {'P_p1': r'$P$', 't0_p1': r'$T_0$', 'p_p1': r'R$_p$/R$_*$',
                   'b_p1': r'$b$', 'q1_SOSS': r'$q_1$', 'q2_SOSS': r'$q_2$',
                   'ecc_p1': r'$e$', 'omega_p1': r'$\Omega$',
                   'sigma_w': r'$\sigma_w_SOSS$',
                   'theta0': r'$\theta_0_SOSS$', 'theta1_SOSS': r'$\theta_1$',
                   'GP_sigma_SOSS': r'$GP_\sigma$', 'GP_rho_SOSS': r'$GP_rho$',
                   'rho': r'$\rho$'}

# Get time axis
t = fits.getdata(outdir + 'lightcurves{}.fits'.format(suffix), 7)
# Noramalized time for trends
tt = np.zeros((280, 1))
tt[:, 0] = (t - np.mean(t)) / np.sqrt(np.var(t))

out_frames = np.abs(out_frames)
out_trans = np.concatenate([np.arange(out_frames[0]),
                            np.arange(out_frames[1]) - out_frames[1]])

for order in orders:

    first_time = True
    outpdf = matplotlib.backends.backend_pdf.PdfPages(outdir + 'lightcurve_fit_order{0}{1}.pdf'.format(order, suffix))

    print('\nFitting order {}\n'.format(order))
    # Unpack wave, flux and error
    wave = fits.getdata(outdir + 'lightcurves{}.fits'.format(suffix),
                        1 + 3 * (order - 1))
    flux = fits.getdata(outdir + 'lightcurves{}.fits'.format(suffix),
                        2 + 3 * (order - 1))
    err = fits.getdata(outdir + 'lightcurves{}.fits'.format(suffix),
                       3 + 3 * (order - 1))
    nints, nbins = np.shape(flux)

    # Set up light curve plots
    data = np.ones((nints, nbins))*np.nan
    models = np.ones((nints, nbins)) * np.nan
    residuals = np.ones((nints, nbins)) * np.nan

    # Set up priors
    priors = {}
    for param, dist, hyperp in zip(params, dists, hyperps):
        priors[param] = {}
        priors[param]['distribution'] = dist
        priors[param]['hyperparameters'] = hyperp

    if order == 1 and nestorprior_o1 is not None:
        print("Porting in Néstor's priors")
        prior_q1, prior_q2 = utils.gen_ldprior_from_nestor(nestorprior_o1, wave[0])
    if order == 2 and nestorprior_o2 is not None:
        print("Porting in Néstor's priors")
        prior_q1, prior_q2 = utils.gen_ldprior_from_nestor(nestorprior_o2, wave[0])

    # Fit each light curve
    outdict = {}
    for i in range(nbins):
        skip = False
        if np.all(flux[:, i] == 0):
            print('skipping bin #{} / {}'.format(i + 1, nbins))
            skip = True
        else:
            print('Fitting bin #{} / {}'.format(i + 1, nbins))
            norm_flux = flux[:, i] / np.median(flux[out_trans, i])
            norm_err = np.zeros_like(err[:, i])

            if nestorprior_o1 is not None or nestorprior_o2 is not None:
                priors['q1_SOSS']['distribution'] = 'truncatednormal'
                priors['q1_SOSS']['hyperparameters'] = [prior_q1[i], 0.1, 0.0, 1.0]
                priors['q2_SOSS']['distribution'] = 'truncatednormal'
                priors['q2_SOSS']['hyperparameters'] = [prior_q2[i], 0.1, 0.0, 1.0]

            dataset = juliet.load(priors=priors, t_lc={'SOSS': t},
                                  y_lc={'SOSS': norm_flux},
                                  yerr_lc={'SOSS': norm_err},
                                  linear_regressors_lc={'SOSS': tt},
                                  out_folder=outdir + 'speclightcurve{2}/order{0}_wavebin{1}'.format(order, i, suffix))
            results = dataset.fit(sampler='dynesty')

        # Pack best fit params into a dictionary
        for param, dist in zip(params, dists):
            if dist == 'fixed':
                continue
            if first_time is True:
                outdict[param + '_m'] = []
                outdict[param + '_u'] = []
                outdict[param + '_l'] = []

            if skip is True:
                outdict[param + '_m'].append(np.nan)
                outdict[param + '_u'].append(np.nan)
                outdict[param + '_l'].append(np.nan)
            else:
                pp = results.posteriors['posterior_samples'][param]
                pm, pu, pl = juliet.utils.get_quantiles(pp)
                outdict[param + '_m'].append(pm)
                outdict[param + '_u'].append(pu)
                outdict[param + '_l'].append(pl)
        first_time = False

        # Make summary plots
        if skip is False:
            transit_model = results.lc.evaluate('SOSS')
            scatter = np.median(results.posteriors['posterior_samples']['sigma_w_SOSS'])
            plotting.do_lightcurve_plot(t=dataset.times_lc['SOSS'],
                                        data=dataset.data_lc['SOSS'],
                                        error=scatter/1e6, model=transit_model,
                                        scatter=scatter, outpdf=outpdf, nfit=5,
                                        title='bin {0} | {1:.3f}µm'.format(i, wave[0, i]))

            fit_params, posterior_names = [], []
            for param, dist in zip(params, dists):
                if dist != 'fixed':
                    fit_params.append(param)
                    if param in formatted_names.keys():
                        posterior_names.append(formatted_names[param])
                    else:
                        posterior_names.append(param)
            plotting.make_corner(fit_params, results, outpdf=outpdf,
                                 posterior_names=posterior_names)

            data[:, i] = norm_flux
            models[:, i] = transit_model
            residuals[:, i] = norm_flux - transit_model

    plotting.plot_2dlightcurves(wave[0], data, outpdf=outpdf,
                                title='Normalized Lightcurves')
    plotting.plot_2dlightcurves(wave[0], models, outpdf=outpdf,
                                title='Model Lightcurves')
    plotting.plot_2dlightcurves(wave[0], residuals, outpdf=outpdf,
                                title='Residuals')
    outdf = pd.DataFrame(data=outdict)
    outdf.to_csv(outdir + 'speclightcurve_results_order{0}{1}.csv'.format(order, suffix),
                 index=False)
    outpdf.close()

print('Done')
