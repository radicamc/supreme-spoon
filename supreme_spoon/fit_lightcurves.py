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
outdir = './'
infile = ''
orders = [1, 2]
suffix = ''
out_frames = [50, -50]

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

ldcoef_file_o1 = None
ldcoef_file_o2 = None
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
t = fits.getdata(infile, 9)
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
    wave_low = fits.getdata(infile,  1 + 4*(order - 1))
    wave_up = fits.getdata(infile, 2 + 4*(order - 1))
    wave = np.nanmean(np.stack([wave_low, wave_up]), axis=0)
    flux = fits.getdata(infile, 3 + 4*(order - 1))
    err = fits.getdata(infile, 4 + 4*(order - 1))
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

    if order == 1 and ldcoef_file_o1 is not None:
        prior_q1, prior_q2 = utils.get_ld_coefs(ldcoef_file_o1, wave_low[0], wave_up[0])
    if order == 2 and ldcoef_file_o2 is not None:
        prior_q1, prior_q2 = utils.get_ld_coefs(ldcoef_file_o2, wave_low[0], wave_up[0])

    # Fit each light curve
    outdict = {}
    for i in range(nbins):
        skip = False
        if not np.isfinite(flux[:, i]).all():
            print('Skipping bin #{} / {}'.format(i + 1, nbins))
            skip = True
        else:
            print('Fitting bin #{} / {}'.format(i + 1, nbins))
            out_med = np.median(flux[out_trans, i])
            norm_flux = flux[:, i] / out_med
            norm_err = np.zeros_like(err[:, i])

            if ldcoef_file_o1 is not None or ldcoef_file_o2 is not None:
                if np.isfinite(prior_q1[i]):
                    priors['q1_SOSS']['distribution'] = 'truncatednormal'
                    priors['q1_SOSS']['hyperparameters'] = [prior_q1[i], 0.1, 0.0, 1.0]
                if np.isfinite(prior_q2[i]):
                    priors['q2_SOSS']['distribution'] = 'truncatednormal'
                    priors['q2_SOSS']['hyperparameters'] = [prior_q2[i], 0.1, 0.0, 1.0]
            try:
                dataset = juliet.load(priors=priors, t_lc={'SOSS': t},
                                      y_lc={'SOSS': norm_flux},
                                      yerr_lc={'SOSS': norm_err},
                                      linear_regressors_lc={'SOSS': tt},
                                      out_folder=outdir + 'speclightcurve{2}/order{0}_wavebin{1}'.format(order, i, suffix))
                results = dataset.fit(sampler='dynesty')
            except KeyboardInterrupt as err:
                raise err
            except:
                print('Exception encountered.')
                print('Skipping bin #{} / {}'.format(i + 1, nbins))
                skip = True

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
            try:
                transit_model = results.lc.evaluate('SOSS')
                scatter = np.median(results.posteriors['posterior_samples']['sigma_w_SOSS'])
                nfit = len(np.where(dists != 'fixed')[0])
                out_dev = np.sqrt(utils.outlier_resistant_variance(norm_flux[out_trans]))
                plotting.do_lightcurve_plot(t=dataset.times_lc['SOSS'],
                                            data=dataset.data_lc['SOSS'],
                                            model=transit_model,
                                            scatter=scatter, out_dev=out_dev,
                                            outpdf=outpdf,
                                            title='bin {0} | {1:.3f}µm'.format(i, wave[0, i]),
                                            nfit=nfit)

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
            except:
                pass

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
