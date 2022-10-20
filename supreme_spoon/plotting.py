#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 14:02 2022

@author: MCR

Plotting routines.
"""

import corner
import matplotlib.backends.backend_pdf
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import warnings


def make_corner_plot(fit_params, results, posterior_names=None, outpdf=None,
                     truths=None):
    """make corner plot for lightcurve fitting.
    """

    first_time = True
    for param in fit_params:
        if first_time:
            pos = results.posteriors['posterior_samples'][param]
            first_time = False
        else:
            pos = np.vstack((pos, results.posteriors['posterior_samples'][param]))

    figure = corner.corner(pos.T, labels=posterior_names, color='black',
                           show_titles=True, title_fmt='.3f',
                           label_kwargs=dict(fontsize=14), truths=truths,
                           facecolor='white')
    if outpdf is not None:
        if isinstance(outpdf, matplotlib.backends.backend_pdf.PdfPages):
            outpdf.savefig(figure)
        else:
            figure.savefig(outpdf)
        figure.clear()
        plt.close(figure)
    else:
        plt.show()


def make_lightcurve_plot(t, data, model, scatter, errors, outpdf=None,
                         title=None, nfit=8):
    """Plot results of lightcurve fit.
    """

    def gaus(x, m, s):
        return np.exp(-0.5 * (x - m)**2 / s**2) / np.sqrt(2 * np.pi * s**2)

    def chi2(o, m, e):
        return np.nansum((o - m)**2 / e**2)

    fig = plt.figure(figsize=(13, 7), facecolor='white')
    gs = GridSpec(4, 1, height_ratios=[3, 1, 0.3, 1])

    # Photometry
    ax1 = plt.subplot(gs[0])
    ax1.errorbar(t, data, yerr=scatter * 1e-6, fmt='o', capsize=0,
                 color='royalblue', ms=5, alpha=1)
    ax1.plot(t, model, color='black', zorder=100)
    ax1.set_ylabel('Relative Flux', fontsize=18)
    ax1.set_xlim(np.min(t), np.max(t))
    ax1.xaxis.set_major_formatter(plt.NullFormatter())
    chi2_v = chi2(data*1e6, model*1e6, errors*1e6) / (len(t) - nfit)
    mean_err = np.nanmean(errors)
    err_mult = scatter / (mean_err*1e6)
    ax1.text(t[2], np.min(model), r'$\chi_\nu^2 = {:.2f}$''\n'r'$\sigma={:.2f}$ppm''\n'r'$e={:.2f}$'.format(chi2_v, mean_err*1e6, err_mult),
             fontsize=14)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)

    if title is not None:
        plt.title(title, fontsize=16)

    # Residuals
    ax2 = plt.subplot(gs[1])
    ax2.errorbar(t, (data - model) * 1e6, yerr=scatter, alpha=1, ms=5,
                 c='royalblue', fmt='o', zorder=10)
    ax2.axhline(0, ls='--', c='black')
    xpos = np.percentile(t, 1)
    plt.text(xpos, np.max((data - model) * 1e6),
             r'{:.2f}$\,$ppm'.format(scatter))
    ax2.fill_between(t, -scatter, scatter, color='black', alpha=0.1)
    ax2.set_xlim(np.min(t), np.max(t))
    ax2.set_ylabel('Residuals\n(ppm)', fontsize=18)
    ax2.set_xlabel('Time from Transit Midpoint [hrs]', fontsize=18)
    ax2.tick_params(axis='x', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)

    # Histogram of residuals
    ax3 = plt.subplot(gs[3])
    res = (data - model) * 1e6 / scatter
    bins = np.linspace(-10, 10, 41) + 0.25
    hist = ax3.hist(res, edgecolor='grey', color='lightgrey', bins=bins)
    area = np.sum(hist[0] * np.diff(bins))
    ax3.plot(np.linspace(-15, 15, 500),
             gaus(np.linspace(-15, 15, 500), 0, 1) * area, c='black')
    ax3.set_ylabel('Counts', fontsize=18)
    ax3.set_xlabel('Residuals/Scatter', fontsize=18)
    ax3.set_xlim(-5, 5)
    ax3.tick_params(axis='x', labelsize=12)
    ax3.tick_params(axis='y', labelsize=12)

    if outpdf is not None:
        if isinstance(outpdf, matplotlib.backends.backend_pdf.PdfPages):
            outpdf.savefig(fig)
        else:
            fig.savefig(outpdf)
        fig.clear()
        plt.close(fig)
    else:
        plt.show()


def make_2d_lightcurve_plot(wave1, flux1, wave2=None, flux2=None, outpdf=None,
                            title='', **kwargs):
    """Plot 2D spectroscopic light curves.
    """

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')

        if 'vmax' not in kwargs:
            kwargs['vmax'] = np.nanpercentile(flux1, 95)
        if 'vmin' not in kwargs:
            kwargs['vmin'] = np.nanpercentile(flux1, 5)

        if title != '':
            title = ': ' + title

        fig = plt.figure(figsize=(12, 5), facecolor='white')
        gs = GridSpec(1, 2, width_ratios=[1, 1])

        ax1 = fig.add_subplot(gs[0, 0])
        pp = ax1.imshow(flux1.T, aspect='auto', origin='lower',
                        extent=(0, flux1.shape[0]-1, wave1[0], wave1[-1]),
                        **kwargs)
        if wave2 is None:
            cax = ax1.inset_axes([1.05, 0.005, 0.03, 0.99],
                                 transform=ax1.transAxes)
            cb = fig.colorbar(pp, ax=ax1, cax=cax)
            cb.set_label('Normalized Flux', labelpad=15, rotation=270,
                         fontsize=16)
        ax1.set_ylabel('Wavelength [µm]', fontsize=16)
        ax1.set_xlabel('Integration Number', fontsize=16)
        plt.title('Order 1' + title, fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        if wave2 is not None:
            ax2 = fig.add_subplot(gs[0, 1])
            pp = ax2.imshow(flux2.T, aspect='auto', origin='lower',
                            extent=(0, flux2.shape[0]-1, wave2[0], wave2[-1]),
                            **kwargs)
            cax = ax2.inset_axes([1.05, 0.005, 0.03, 0.99],
                                 transform=ax2.transAxes)
            cb = fig.colorbar(pp, ax=ax2, cax=cax)
            cb.set_label('Normalized Flux', labelpad=15, rotation=270,
                         fontsize=16)
            ax2.set_xlabel('Integration Number', fontsize=16)
            plt.title('Order 2' + title, fontsize=18)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)

            gs.update(wspace=0.1)

    if outpdf is not None:
        if isinstance(outpdf, matplotlib.backends.backend_pdf.PdfPages):
            outpdf.savefig(fig)
        else:
            fig.savefig(outpdf)
        fig.clear()
        plt.close(fig)
    else:
        plt.show()
