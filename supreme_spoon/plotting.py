#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 14:02 2022

@author: MCR

Plotting routines.
"""

from astropy.timeseries import LombScargle
import bottleneck as bn
import corner
import matplotlib.backends.backend_pdf
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import warnings

from supreme_spoon import utils


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


def make_lightcurve_plot(t, data, model, scatter, errors, nfit, outpdf=None,
                         title=None, systematics=None, rasterized=False,
                         nbin=10):
    """Plot results of lightcurve fit.
    """

    def gaus(x, m, s):
        return np.exp(-0.5*(x - m)**2/s**2)/np.sqrt(2*np.pi*s**2)

    def chi2(o, m, e):
        return np.nansum((o - m)**2/e**2)

    if systematics is not None:
        fig = plt.figure(figsize=(13, 9), facecolor='white',
                         rasterized=rasterized)
        gs = GridSpec(5, 1, height_ratios=[3, 3, 1, 0.3, 1])
    else:
        fig = plt.figure(figsize=(13, 7), facecolor='white',
                         rasterized=rasterized)
        gs = GridSpec(4, 1, height_ratios=[3, 1, 0.3, 1])

    # Light curve with full systematics + astrophysical model.
    ax1 = plt.subplot(gs[0])
    assert len(data) == len(model)
    nint = len(data)  # Total number of data points
    # Full dataset
    ax1.errorbar(t, data, yerr=scatter*1e-6, fmt='o', capsize=0,
                 color='royalblue', ms=5, alpha=0.75)
    # Binned points
    rem = nint % nbin
    if rem != 0:
        trim_i = np.random.randint(0, rem)
        trim_e = -1*(rem-trim_i)
        t_bin = t[trim_i:trim_e].reshape((nint-rem)//nbin, nbin)
        d_bin = data[trim_i:trim_e].reshape((nint-rem)//nbin, nbin)
    else:
        t_bin = t.reshape((nint-rem)//nbin, nbin)
        d_bin = data.reshape((nint-rem)//nbin, nbin)
    t_bin = np.nanmean(t_bin, axis=1)
    d_bin = np.nanmean(d_bin, axis=1)
    ax1.errorbar(t_bin, d_bin, yerr=scatter*1e-6/np.sqrt(nbin), fmt='o',
                 mfc='blue', mec='white', ecolor='blue', ms=8, alpha=1,
                 zorder=11)
    # Other stuff.
    ax1.plot(t, model, color='black', zorder=10)
    ax1.set_ylabel('Relative Flux', fontsize=18)
    ax1.set_xlim(np.min(t), np.max(t))
    ax1.xaxis.set_major_formatter(plt.NullFormatter())
    chi2_v = chi2(data*1e6, model*1e6, errors*1e6) / (len(t) - nfit)
    mean_err = np.nanmean(errors)
    err_mult = scatter / (mean_err*1e6)
    ax1.text(t[2], np.min(model),
             r'$\chi_\nu^2 = {:.2f}$''\n'r'$\sigma={:.2f}$ppm''\n'r'$e={:.2f}$'.format(
                 chi2_v, mean_err*1e6, err_mult),
             fontsize=14)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)

    if title is not None:
        plt.title(title, fontsize=16)

    # Detrended Light curve.
    if systematics is not None:
        ax2 = plt.subplot(gs[1])
        assert len(model) == len(systematics)
        model_detrended = model - systematics
        data_detrended = data - systematics
        # Full dataset.
        ax2.errorbar(t, data_detrended, yerr=scatter*1e-6, fmt='o',
                     capsize=0, color='salmon', ms=5, alpha=1)
        # Binned points.
        if rem != 0:
            d_bin = data_detrended[trim_i:trim_e].reshape((nint-rem)//nbin, nbin)
        else:
            d_bin = data_detrended.reshape((nint-rem)//nbin, nbin)
        d_bin = np.nanmean(d_bin, axis=1)
        ax2.errorbar(t_bin, d_bin, yerr=scatter*1e-6/np.sqrt(nbin), fmt='o',
                     mfc='red', mec='white', ecolor='red', ms=8, alpha=1,
                     zorder=11)
        # Other stuff.
        ax2.plot(t, model_detrended, color='black', zorder=10)
        ax2.set_ylabel('Relative Flux\n(Detrended)', fontsize=18)
        ax2.set_xlim(np.min(t), np.max(t))
        ax2.xaxis.set_major_formatter(plt.NullFormatter())
        ax2.tick_params(axis='x', labelsize=12)
        ax2.tick_params(axis='y', labelsize=12)

    # Residuals.
    if systematics is not None:
        ax3 = plt.subplot(gs[2])
    else:
        ax3 = plt.subplot(gs[1])
    # Full dataset.
    res = (data - model)*1e6
    ax3.errorbar(t, res, yerr=scatter, alpha=0.8, ms=5,
                 c='royalblue', fmt='o', zorder=10)
    # Binned points.
    if rem != 0:
        r_bin = res[trim_i:trim_e].reshape((nint-rem)//nbin, nbin)
    else:
        r_bin = res.reshape((nint-rem)//nbin, nbin)
    r_bin = np.nanmean(r_bin, axis=1)
    ax3.errorbar(t_bin, r_bin, yerr=scatter/np.sqrt(nbin), fmt='o',
                 mfc='blue', mec='white', ecolor='blue', ms=8, alpha=1,
                 zorder=11)
    # Other stuff.
    ax3.axhline(0, ls='--', c='black')
    xpos = np.percentile(t, 1)
    plt.text(xpos, np.max((data - model)*1e6),
             r'{:.2f}$\,$ppm'.format(scatter))
    ax3.fill_between(t, -scatter, scatter, color='black', alpha=0.1)
    ax3.set_xlim(np.min(t), np.max(t))
    ax3.set_ylabel('Residuals\n(ppm)', fontsize=18)
    ax3.set_xlabel('Time from Transit Midpoint [hrs]', fontsize=18)
    ax3.tick_params(axis='x', labelsize=12)
    ax3.tick_params(axis='y', labelsize=12)

    # Histogram of residuals.
    if systematics is not None:
        ax4 = plt.subplot(gs[4])
    else:
        ax4 = plt.subplot(gs[3])
    bins = np.linspace(-10, 10, 41) + 0.25
    hist = ax4.hist(res/scatter, edgecolor='grey', color='lightgrey', bins=bins)
    area = np.sum(hist[0] * np.diff(bins))
    ax4.plot(np.linspace(-15, 15, 500),
             gaus(np.linspace(-15, 15, 500), 0, 1) * area, c='black')
    ax4.set_ylabel('Counts', fontsize=18)
    ax4.set_xlabel('Residuals/Scatter', fontsize=18)
    ax4.set_xlim(-5, 5)
    ax4.tick_params(axis='x', labelsize=12)
    ax4.tick_params(axis='y', labelsize=12)

    if outpdf is not None:
        if isinstance(outpdf, matplotlib.backends.backend_pdf.PdfPages):
            outpdf.savefig(fig)
        else:
            fig.savefig(outpdf)
        fig.clear()
        plt.close(fig)
    else:
        plt.show()


def make_linearity_plot(cube, old_cube):
    """Plot group differences before and after linearity correction.
    """

    nint, ngroup, dimy, dimx = np.shape(cube)
    # Get bright pixels in the trace.
    stack = bn.nanmedian(cube[np.random.randint(0, nint, 25), -1], axis=0)
    ii = np.where((stack >= np.nanpercentile(stack, 80)) &
                  (stack < np.nanpercentile(stack, 99)))

    # Compute group differences in these pixels.
    new_diffs = np.zeros((ngroup-1, len(ii[0])))
    old_diffs = np.zeros((ngroup-1, len(ii[0])))
    num_pix = 20000
    if len(ii[0]) < 20000:
        num_pix = len(ii[0])
    for it in tqdm(range(num_pix)):
        ypos, xpos = ii[0][it], ii[1][it]
        # Choose a random integration.
        i = np.random.randint(0, nint)
        # Calculate the group differences.
        new_diffs[:, it] = np.diff(cube[i, :, ypos, xpos])
        old_diffs[:, it] = np.diff(old_cube[i, :, ypos, xpos])

    new_med = np.mean(new_diffs, axis=1)
    old_med = np.mean(old_diffs, axis=1)

    # Plot up mean group differences before and after linearity correction.
    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(len(new_med)), new_med - np.mean(new_med),
             label='After Correction', c='blue', lw=2)
    plt.plot(np.arange(len(new_med)), old_med - np.mean(old_med),
             label='Before Correction', c='red', lw=2)
    plt.axhline(0, ls='--', c='black', zorder=0)
    plt.xlabel(r'Groups', fontsize=12)
    locs = np.arange(ngroup-1).astype(int)
    labels = []
    for i in range(ngroup-1):
        labels.append('{0}-{1}'.format(i+2, i+1))
    plt.xticks(locs, labels, rotation=45)
    plt.ylabel('Differences [DN]', fontsize=12)
    plt.ylim(1.1*np.min(old_med - np.mean(old_med)),
             1.1*np.max(old_med - np.mean(old_med)))
    plt.legend()
    plt.show()


def make_oneoverf_psd(cube, old_cube, timeseries, baseline_ints, nsample=100,
                      mask_cube=None, occultation_type='transit',
                      tframe=5.494, tpix=1e-5, tgap=1.2e-4):
    """Make a PSD plot to see PSD of background before and after 1/f removal.
    """

    nints, ngroups, dimy, dimx = np.shape(cube)
    baseline_ints = utils.format_out_frames(baseline_ints, occultation_type)
    old_deep = bn.nanmedian(old_cube[baseline_ints], axis=0)
    deep = bn.nanmedian(cube[baseline_ints], axis=0)

    # Generate array of timestamps for each pixel
    pixel_ts = []
    time1 = 0
    for p in range(dimy * dimx):
        ti = time1 + tpix
        # If column is done, add gap time.
        if p % 256 == 0 and p != 0:
            ti += tgap
        pixel_ts.append(ti)
        time1 = ti

    # Generate psd frequency array
    freqs = np.logspace(np.log10(1 / tframe), np.log10(1 / tpix), 100)
    pwr_old = np.zeros((nsample, len(freqs)))
    pwr = np.zeros((nsample, len(freqs)))
    # Select nsample random frames and compare PSDs before and after 1/f
    # removal.
    for s in tqdm(range(nsample)):
        # Get random groups and ints
        i, g = np.random.randint(nints), np.random.randint(ngroups)
        # Get difference images before and after 1/f removal.
        diff_old = (old_cube[i, g] - old_deep[g] * timeseries[i]).flatten('F')[::-1]
        diff = (cube[i, g] - deep[g] * timeseries[i]).flatten('F')[::-1]
        # Mask pixels which are not part of the background
        if mask_cube is None:
            # If no pixel/trace mask, discount pixels above a threshold.
            bad = np.where(np.abs(diff) > 100)
        else:
            # Mask flagged pixels.
            bad = np.where(mask_cube[i, g] != 0)
        diff, diff_old = np.delete(diff, bad), np.delete(diff_old, bad)
        this_t = np.delete(pixel_ts, bad)
        # Calculate PSDs
        pwr_old[s] = LombScargle(this_t, diff_old).power(freqs, normalization='psd')
        pwr[s] = LombScargle(this_t, diff).power(freqs, normalization='psd')

    # Generate the approximate purely white noise level.
    rndm = np.random.normal(0, np.nanstd(diff), len(diff))
    wht = LombScargle(this_t, rndm).power(freqs, normalization='psd')

    # Make the plot.
    plt.figure(figsize=(7, 3))
    # Individual power series.
    for i in range(nsample):
        plt.plot(freqs[:-1], pwr_old[i, :-1], c='salmon', alpha=0.1)
        plt.plot(freqs[:-1], pwr[i, :-1], c='royalblue', alpha=0.1)
    # Median trends.
    # Aprox white noise level
    plt.plot(freqs[:-1], np.median(pwr_old, axis=0)[:-1], c='red', lw=2,
             label='Before Correction')
    plt.plot(freqs[:-1], np.median(pwr, axis=0)[:-1], c='blue', lw=2,
             label='After Correction')

    plt.xscale('log')
    plt.xlabel('Frequency [Hz]', fontsize=12)
    plt.yscale('log')
    plt.ylim(np.percentile(pwr, 0.1), np.max(pwr_old))
    plt.ylabel('PSD', fontsize=12)
    plt.legend(loc=1)
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
