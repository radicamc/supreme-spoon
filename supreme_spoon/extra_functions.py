#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 15:08 2023

@author: MCR

Extra functions for use alongside the main pipeline.
"""

from astropy.io import fits
from astroquery.mast import Observations
import bottleneck as bn
from itertools import groupby
import numpy as np
import os
import ray
from scipy.interpolate import interp2d
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from scipy.signal import medfilt
import shutil

from supreme_spoon import utils
from supreme_spoon.utils import fancyprint


def download_observations(proposal_id, instrument_name=None, objectname=None,
                          filters=None, visit_nos=None):
    """Directly download uncal files associated with a given observation from
    the MAST archive.

    Parameters
    ----------
    proposal_id : str
        ID for proposal with which the observations are associated.
    instrument_name : str
        Name of instrument data to retrieve. NIRISS/SOSS, NIRSPEC/SLIT
        currently supported. (optional).
    objectname : str
        Name of observational target. (optional).
    filters : str
        Instrument filters to retrieve.
    visit_nos : int, array-like(int)
        For targets with multiple visits, which visits to retrieve.
    """

    # Make sure something is specified to download.
    if proposal_id is None and objectname is None:
        msg = 'At least one of proposal_id or objectname must be specified.'
        raise ValueError(msg)

    # Get observations from MAST.
    obs_table = Observations.query_criteria(proposal_id=proposal_id,
                                            instrument_name=instrument_name,
                                            filters=filters,
                                            objectname=objectname,
                                            radius='.2 deg')
    all_products = Observations.get_product_list(obs_table)

    products = Observations.filter_products(all_products,
                                            dataproduct_type='image',
                                            extension='_uncal.fits',
                                            productType='SCIENCE')

    # If specific visits are specified, retrieve only those files. If not,
    # retrieve all relevant files.
    if visit_nos is not None:
        # Group files by observation number.
        nums = []
        for file in products['productFilename'].data.data:
            nums.append(file.split('_')[0])
        nums = np.array(nums)
        exps = [list(j) for i, j in groupby(nums)]
        fancyprint('Identified {} observations.'.format(len(exps)))

        # Select only files from relevant visits.
        visit_nos = np.atleast_1d(visit_nos)
        if np.max(visit_nos) > len(exps):
            msg = 'You are trying to retrieve visit {0}, but only {1} ' \
                  'visits exist.'.format(np.max(visit_nos), len(exps))
            raise ValueError(msg)
        fancyprint('Retrieving visit(s) {}.'.format(visit_nos))
        for visit in visit_nos:
            ii = np.where(nums == exps[visit - 1][0])[0]
            this_visit = products[ii]
            # Download the relevant files.
            manifest = Observations.download_products(this_visit)
    else:
        # Download the relevant files.
        manifest = Observations.download_products(products)

    # Unpack auto-generated directories into something better.
    os.mkdir('DMS_uncal')
    for root, _, files in os.walk('mastDownload/', topdown=False):
        for name in files:
            file = os.path.join(root, name)
            shutil.move(file, 'DMS_uncal/.')
    shutil.rmtree('mastDownload/')

    return


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


def make_smoothed_2d_lightcurve(spec, baseline_ints, nint, dimx, filename,
                                order=1, tscale=3, wscale=9):
    """Smooth extracted 2D SOSS light curves on specified time and wavelength
    scales to use as input for chromatic 1/f correction.

    Parameters
    ----------
    spec : array-like(float)
        Extracted 2D light curves.
    baseline_ints : int, array-like(int)
        Integrations or ingress and/or egress.
    nint : int
        Number of integration in exposure.
    dimx : int
        Number of wavelength bins in exposure.
    filename : str
        File to which to save results.
    order : int
        SOSS diffraction order being considered.
    tscale : int
        Timescale, in integrations, on which to smooth. Must be odd.
    wscale : int
        Timescale, in wavelength bins, on which to smooth. Must be odd.
    """

    # Normalize light curves.
    baseline_ints = utils.format_out_frames(baseline_ints)
    spec /= np.nanmedian(spec[baseline_ints], axis=0)

    # Smooth on desired scale.
    spec_smoothed = medfilt(spec, (tscale, wscale))

    # Put back on full size wavelength axis.
    ref_file = np.ones((nint, dimx))
    if order == 1:
        ref_file[:, 4:-4] = spec_smoothed
    else:
        ref_file[:, 1206:1770] = spec_smoothed

    # Save file.
    suffix = 'lcestimate_2d_o{}.npy'.format(order)
    np.save(filename + suffix, ref_file)


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
