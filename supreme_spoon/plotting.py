#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 14:02 2022

@author: MCR

JWST data analysis plotting routines.
"""

import corner
import matplotlib.backends.backend_pdf
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import warnings


def plot_2dlightcurves(flux1, flux2, wave1, wave2, savename=None, **kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')

        if 'vmax' not in kwargs:
            kwargs['vmax'] = np.nanpercentile(flux1, 95)
        if 'vmin' not in kwargs:
            kwargs['vmin'] = np.nanpercentile(flux1, 5)

        fig = plt.figure(figsize=(12, 5), facecolor='white')
        gs = GridSpec(1, 2, width_ratios=[1, 1])

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(flux1.T, aspect='auto', origin='lower',
                   extent=(0, flux1.shape[0]-1, wave1[0], wave1[-1]), **kwargs)
        ax1.set_ylabel('Wavelength [µm]', fontsize=16)
        ax1.set_xlabel('Integration Number', fontsize=16)
        plt.title('Order 1', fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        ax2 = fig.add_subplot(gs[0, 1])
        pp = ax2.imshow(flux2.T, aspect='auto', origin='lower',
                        extent=(0, flux2.shape[0]-1, wave2[0], wave2[-1]), **kwargs)
        cax = ax2.inset_axes([1.05, 0.005, 0.03, 0.99],
                             transform=ax2.transAxes)
        cb = fig.colorbar(pp, ax=ax2, cax=cax)
        cb.set_label('Normalized Flux', labelpad=15, rotation=270, fontsize=16)
        ax2.set_xlabel('Integration Number', fontsize=16)
        plt.title('Order 2', fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        gs.update(wspace=0.1)

    if savename is not None:
        plt.savefig(savename, format='pdf')
    plt.show()


def do_lightcurve_plot(t, data, error, model, sigma, scatter, uperr=None,
                       lerr=None, outpdf=None,
                       title=None, nfit=8):
    def gaus(x, m, s):
        return np.exp(-0.5 * (x - m) ** 2 / s ** 2) / np.sqrt(
            2 * np.pi * s ** 2)

    def chi2(o, m, e):
        return np.nansum((o - m) ** 2 / e ** 2)

    fig = plt.figure(figsize=(13, 7), facecolor='white')
    gs = GridSpec(4, 1, height_ratios=[3, 1, 0.3, 1])

    # Photometry
    ax1 = plt.subplot(gs[0])
    ax1.errorbar(t, data, yerr=error, fmt='o', capsize=0,
                 color='royalblue', ms=5, alpha=1)
    ax1.plot(t, model, color='black', zorder=100)
    ax1.set_ylabel('Relative Flux', fontsize=14)
    ax1.set_xlim(np.min(t), np.max(t))
    ax1.xaxis.set_major_formatter(plt.NullFormatter())
    chi2_v = chi2(data * 1e6, model * 1e6, sigma) / (len(t) - nfit)
    ax1.text(t[2], np.min(model), r'$\chi_\nu^2 = {:.2f}$'.format(chi2_v),
             fontsize=14)

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
    ax2.fill_between(t, -sigma, sigma, color='black', alpha=0.1)
    ax2.set_xlim(np.min(t), np.max(t))
    ax2.set_ylabel('Residuals\n(ppm)', fontsize=16)
    ax2.set_xlabel('Time [BJD]', fontsize=16)

    # Histogram of residuals
    ax3 = plt.subplot(gs[3])
    res = (data - model) * 1e6 / sigma
    bins = np.linspace(-10, 10, 41) + 0.25
    hist = ax3.hist(res, edgecolor='grey', color='lightgrey', bins=bins)
    area = np.sum(hist[0] * np.diff(bins))
    ax3.plot(np.linspace(-15, 15, 500),
             gaus(np.linspace(-15, 15, 500), 0, 1) * area, c='black')
    ax3.set_ylabel('Counts', fontsize=16)
    ax3.set_xlabel('Residuals (sigma)', fontsize=16)

    ii = np.where(hist[0] != 0)
    start = hist[1][np.min(ii)] - 1
    end = hist[1][np.max(ii)] + 2
    ax3.set_xlim(start, end)

    if outpdf is not None:
        if isinstance(outpdf, matplotlib.backends.backend_pdf.PdfPages):
            outpdf.savefig(fig)
        else:
            fig.savefig(outpdf)
        plt.close()
    else:
        plt.show()


def make_corner(fit_params, results, posterior_names=None, outpdf=None,
                truths=None):
    first_time = True
    for param in fit_params:
        if first_time:
            pos = results.posteriors['posterior_samples'][param]
            first_time = False
        else:
            pos = np.vstack(
                (pos, results.posteriors['posterior_samples'][param]))

    figure = corner.corner(pos.T, labels=posterior_names, color='black',
                           show_titles=True,
                           title_fmt='.3f', label_kwargs=dict(fontsize=14),
                           truths=truths,
                           facecolor='white')
    if outpdf is not None:
        if isinstance(outpdf, matplotlib.backends.backend_pdf.PdfPages):
            outpdf.savefig(figure)
        else:
            figure.savefig(outpdf)
        plt.close()
    else:
        plt.show()

