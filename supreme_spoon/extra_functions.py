#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 15:08 2023

@author: MCR

Extra functions to be used alongside the main pipeline.
"""

from astropy.io import fits
import numpy as np


def get_throughput_from_photom_file(photom_path):
    """Calculate the throughput based on the photom reference file.
    Function from Loïc, and is apparently the proper way to get the
    throughput? Gives different results from contents of spectra reference
    file.

    Parameters
    ----------
    photom_path : str
        Path to photom reference file

    Returns
    -------
    w1 : np.array(float)
        Order 1 wavelength axis.
    w2 : np.array(float)
        Order 2 wavelength axis.
    thpt1 : np.array(float)
        Order 1 throughput values
    thpt2 : np.array(float)
        Order 2 throughput values
    """

    # From the photom ref file, get conversion factors + wavelength/pixel grid.
    photom = fits.open(photom_path)
    w1 = photom[1].data['wavelength'][0]
    ii = np.where((photom[1].data['wavelength'][0] >= 0.84649785) &
                  (photom[1].data['wavelength'][0] <= 2.83358154))
    w1 = w1[ii]
    scale1 = photom[1].data['relresponse'][0][ii]

    w2 = photom[1].data['wavelength'][1]
    ii = np.where((photom[1].data['wavelength'][1] >= 0.49996997) &
                  (photom[1].data['wavelength'][1] <= 1.40884607))
    w2 = w2[ii]
    scale2 = photom[1].data['relresponse'][1][ii]

    # Calculate throughput from conversion factor.
    thpt1 = 1 / (scale1 * 3e8 / w1 ** 2)
    thpt2 = 1 / (scale2 * 3e8 / w2 ** 2)

    return w1, w2, thpt1, thpt2


# ====== Deprecated CCF Stability Functions --- May Have Later Uses?? =====
def soss_stability_xy(cube, nsteps=501, axis='x', nthreads=4,
                      smoothing_scale=None):
    """Perform a CCF analysis to track the movement of the SOSS trace
        relative to the median stack over the course of a TSO.
    Parameters
    ----------
    cube : array-like[float]
        Data cube. Should be 3D (ints, dimy, dimx).
    nsteps : int
        Number of CCF steps to test.
    axis : str
        Axis over which to calculate the CCF - either 'x', or 'y'.
    nthreads : int
        Number of CPUs for multiprocessing.
    smoothing_scale : int
        Length scale over which to smooth results.
    Returns
    -------
    ccf : array-like[float]
        The cross-correlation results.
    """
    # Initialize ray with specified number of threads.
    ray.shutdown()
    ray.init(num_cpus=nthreads)
    # Subtract integration-wise median from cube for CCF.
    cube = cube - np.nanmedian(cube, axis=(1, 2))[:, None, None]
    # Calculate median stack.
    deep = bn.nanmedian(cube, axis=0)
    # Divide total data cube into segments and run each segment in parallel
    # with ray.
    ii = 0
    all_fits = []
    nints = np.shape(cube)[0]
    seglen = nints // nthreads
    for i in range(nthreads):
        if i == nthreads - 1:
            cube_seg = cube[ii:]
        else:
            cube_seg = cube[ii:ii + seglen]
        all_fits.append(soss_stability_xy_run.remote(cube_seg, deep, seg_no=i+1,
                                                     nsteps=nsteps, axis=axis))
        ii += seglen
    # Run the CCFs.
    ray_results = ray.get(all_fits)
    # Stack all the CCF results into a single array.
    maxvals = []
    for i in range(nthreads):
        if i == 0:
            maxvals = ray_results[i]
        else:
            maxvals = np.concatenate([maxvals, ray_results[i]])
    # Smooth results if requested.
    if smoothing_scale is not None:
        ccf = median_filter(np.linspace(-0.01, 0.01, nsteps)[maxvals],
                            smoothing_scale)
    else:
        ccf = np.linspace(-0.01, 0.01, nsteps)[maxvals]
    ccf = ccf.reshape(nints)
    return ccf


@ray.remote
def soss_stability_xy_run(cube_sub, med, seg_no, nsteps=501, axis='x'):
    """Wrapper to perform CCF calculations in parallel with ray.
    """
    # Get data dimensions.
    nints, dimy, dimx = np.shape(cube_sub)
    # Get integration numbers to show progress prints.
    marks = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    locs = np.nanpercentile(np.arange(nints), marks)
    # Initialize CCF variables.
    ccf = np.zeros((nints, nsteps))
    f = interp2d(np.arange(dimx), np.arange(dimy), med, kind='cubic')
    # Perform cross-correlation over desired axis.
    loc = 0
    for i in range(nints):
        # Progress print.
        if i >= int(locs[loc]):
            fancyprint('Slice {}: {}% complete.'.format(seg_no, marks[loc]))
            loc += 1
        for j, jj in enumerate(np.linspace(-0.01, 0.01, nsteps)):
            if axis == 'x':
                interp = f(np.arange(dimx) + jj, np.arange(dimy))
            elif axis == 'y':
                interp = f(np.arange(dimx), np.arange(dimy) + jj)
            else:
                raise ValueError('Unknown axis: {}'.format(axis))
            ccf[i, j] = np.nansum(cube_sub[i] * interp)
    # Determine the peak of the CCF for each integration to get the
    # best-fitting shift.
    maxvals = []
    for i in range(nints):
        maxvals.append(np.where(ccf[i] == np.max(ccf[i]))[0])
    maxvals = np.array(maxvals)
    maxvals = maxvals.reshape(maxvals.shape[0])
    return maxvals


def soss_stability_fwhm(cube, ycens_o1, nthreads=4, smoothing_scale=None):
    """Estimate the FWHM of the trace over the course of a TSO by fitting a
    Gaussian to each detector column.
    Parameters
    ----------
    cube : array-like[float]
        Data cube. Should be 3D (ints, dimy, dimx).
    ycens_o1 : arrray-like[float]
        Y-centroid positions of the order 1 trace. Should have length dimx.
    nthreads : int
        Number of CPUs for multiprocessing.
    smoothing_scale : int
        Length scale over which to smooth results.
    Returns
    -------
    fwhm : array-like[float]
        FWHM estimates for each column at every integration.
    """
    # Initialize ray with specified number of threads.
    ray.shutdown()
    ray.init(num_cpus=nthreads)
    # Divide total data cube into segments and run each segment in parallel
    # with ray.
    ii = 0
    all_fits = []
    nints = np.shape(cube)[0]
    seglen = nints // nthreads
    for i in range(nthreads):
        if i == nthreads - 1:
            cube_seg = cube[ii:]
        else:
            cube_seg = cube[ii:ii + seglen]
        all_fits.append(soss_stability_fwhm_run.remote(cube_seg, ycens_o1,
                                                       seg_no=i+1))
        ii += seglen
    # Run the CCFs.
    ray_results = ray.get(all_fits)
    # Stack all the CCF results into a single array.
    fwhm = []
    for i in range(nthreads):
        if i == 0:
            fwhm = ray_results[i]
        else:
            fwhm = np.concatenate([fwhm, ray_results[i]])
    # Set median of trend to zero.
    fwhm -= np.median(fwhm)
    # Smooth the trend.
    if smoothing_scale is None:
        smoothing_scale = int(0.2*nints)
    fwhm = median_filter(fwhm, smoothing_scale)
    return fwhm


@ray.remote
def soss_stability_fwhm_run(cube, ycens_o1, seg_no):
    """Wrapper to perform FWHM calculations in parallel with ray.
    """
    def gauss(x, *p):
        amp, mu, sigma = p
        return amp * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))
    # Get data dimensions.
    nints, dimy, dimx = np.shape(cube)
    # Get integration numbers to show progress prints.
    marks = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    locs = np.nanpercentile(np.arange(nints), marks)
    # Initialize storage array for widths.
    fwhm = np.zeros((nints, dimx-254))
    # Fit a Gaussian to the PSF in each detector column.
    loc = 0
    for j in range(nints):
        # Progress print.
        if j >= int(locs[loc]):
            fancyprint('Slice {}: {}% complete.'.format(seg_no, marks[loc]))
            loc += 1
        # Cut out first 250 columns as there is order 2 contmination.
        for i in range(250, dimx-4):
            p0 = [1., ycens_o1[i], 1.]
            data = np.copy(cube[j, :, i])
            # Replace any NaN values with a median.
            if np.isnan(data).any():
                ii = np.where(np.isnan(data))
                data[ii] = np.nanmedian(data)
            # Fit a Gaussian to the profile, and save the FWHM.
            try:
                coeff, var_matrix = curve_fit(gauss, np.arange(dimy), data,
                                              p0=p0)
                fwhm[j, i-250] = coeff[2] * 2.355
            except RuntimeError:
                fwhm[j, i-250] = np.nan
    # Get median FWHM per integration.
    fwhm = np.nanmedian(fwhm, axis=1)
    return fwhm
