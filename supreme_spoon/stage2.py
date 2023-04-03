#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Jul 21 17:33 2022

@author: MCR

Custom JWST DMS pipeline steps for Stage 2 (Spectroscopic processing).
"""

from astropy.io import fits
import glob
import numpy as np
import os
import pandas as pd
from scipy.ndimage import median_filter
from tqdm import tqdm
import warnings

from jwst import datamodels
from jwst.extract_1d.soss_extract import soss_boxextract
from jwst.pipeline import calwebb_spec2

from supreme_spoon import utils
from supreme_spoon.utils import fancyprint


class AssignWCSStep:
    """Wrapper around default calwebb_spec2 Assign WCS step.
    """

    def __init__(self, datafiles, output_dir='./'):
        """Step initializer.
        """

        self.tag = 'assignwcsstep.fits'
        self.output_dir = output_dir
        self.datafiles = np.atleast_1d(datafiles)
        self.fileroots = utils.get_filename_root(self.datafiles)

    def run(self, save_results=True, force_redo=False, **kwargs):
        """Method to run the step.
        """

        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                fancyprint('File {} already exists.'.format(expected_file))
                fancyprint('Skipping Assign WCS Step.\n')
                res = expected_file
            # If no output files are detected, run the step.
            else:
                step = calwebb_spec2.assign_wcs_step.AssignWcsStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results, **kwargs)
            results.append(res)

        return results


class SourceTypeStep:
    """Wrapper around default calwebb_spec2 Source Type Determination step.
    """

    def __init__(self, datafiles, output_dir='./'):
        """Step initializer.
        """

        self.tag = 'sourcetypestep.fits'
        self.output_dir = output_dir
        self.datafiles = np.atleast_1d(datafiles)
        self.fileroots = utils.get_filename_root(self.datafiles)

    def run(self, save_results=True, force_redo=False, **kwargs):
        """Method to run the step.
        """

        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                fancyprint('File {} already exists.'.format(expected_file))
                fancyprint('Skipping Source Type Determination Step.\n')
                res = expected_file
            # If no output files are detected, run the step.
            else:
                step = calwebb_spec2.srctype_step.SourceTypeStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results, **kwargs)
            results.append(res)

        return results


class BackgroundStep:
    """Wrapper around custom Background Subtraction step.
    """

    def __init__(self, input_data, background_model, output_dir='./'):
        """Step initializer.
        """

        self.tag = 'backgroundstep.fits'
        self.background_model = background_model
        self.output_dir = output_dir
        self.datafiles = utils.sort_datamodels(input_data)
        self.fileroots = utils.get_filename_root(self.datafiles)
        self.fileroot_noseg = utils.get_filename_root_noseg(self.fileroots)

    def run(self, save_results=True, force_redo=False, **kwargs):
        """Method to run the step.
        """

        all_files = glob.glob(self.output_dir + '*')
        do_step = 1
        results, background_models = [], []
        for i in range(len(self.datafiles)):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            expected_bkg = self.output_dir + self.fileroot_noseg + 'background.npy'
            if expected_file not in all_files or expected_bkg not in all_files:
                do_step = 0
                break
            else:
                results.append(datamodels.open(expected_file))
                background_models.append(np.load(expected_bkg))
        if do_step == 1 and force_redo is False:
            fancyprint('Output files already exist.')
            fancyprint('Skipping Background Subtraction Step.')
        # If no output files are detected, run the step.
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                scale1, scale2 = None, None
                if 'scale1' in kwargs.keys():
                    scale1 = kwargs['scale1']
                if 'scale2' in kwargs.keys():
                    scale2 = kwargs['scale2']
                step_results = backgroundstep(self.datafiles,
                                              self.background_model,
                                              output_dir=self.output_dir,
                                              save_results=save_results,
                                              fileroots=self.fileroots,
                                              fileroot_noseg=self.fileroot_noseg,
                                              scale1=scale1, scale2=scale2)
                results, background_models = step_results

        return results, background_models


class FlatFieldStep:
    """Wrapper around default calwebb_spec2 Flat Field Correction step.
    """

    def __init__(self, datafiles, output_dir='./'):
        """Step initializer.
        """

        self.tag = 'flatfieldstep.fits'
        self.output_dir = output_dir
        self.datafiles = np.atleast_1d(datafiles)
        self.fileroots = utils.get_filename_root(self.datafiles)

    def run(self, save_results=True, force_redo=False, **kwargs):
        """Method to run the step.
        """

        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                fancyprint('File {} already exists.'.format(expected_file))
                fancyprint('Skipping Flat Field Correction Step.\n')
                res = expected_file
            # If no output files are detected, run the step.
            else:
                step = calwebb_spec2.flat_field_step.FlatFieldStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results, **kwargs)
            results.append(res)

        return results


class BadPixStep:
    """Wrapper around custom Bad Pixel Correction Step.
    """

    def __init__(self, input_data, smoothed_wlc, baseline_ints,
                 output_dir='./', occultation_type='transit'):
        """Step initializer.
        """

        self.tag = 'badpixstep.fits'
        self.output_dir = output_dir
        self.smoothed_wlc = smoothed_wlc
        self.baseline_ints = baseline_ints
        self.occultation_type = occultation_type
        self.datafiles = utils.sort_datamodels(input_data)
        self.fileroots = utils.get_filename_root(self.datafiles)
        self.fileroot_noseg = utils.get_filename_root_noseg(self.fileroots)

    def run(self, thresh=10, box_size=5, max_iter=1, save_results=True,
            force_redo=False):
        """Method to run the step.
        """

        all_files = glob.glob(self.output_dir + '*')
        do_step = 1
        results = []
        for i in range(len(self.datafiles)):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            expected_deep = self.output_dir + self.fileroot_noseg + 'deepframe.fits'
            if expected_file not in all_files or expected_deep not in all_files:
                do_step = 0
                break
            else:
                results.append(datamodels.open(expected_file))
                deepframe = fits.getdata(expected_deep)
        if do_step == 1 and force_redo is False:
            fancyprint('Output files already exist.')
            fancyprint('Skipping Bad Pixel Correction Step.\n')
        # If no output files are detected, run the step.
        else:
            step_results = badpixstep(self.datafiles,
                                      baseline_ints=self.baseline_ints,
                                      smoothed_wlc=self.smoothed_wlc,
                                      output_dir=self.output_dir,
                                      save_results=save_results,
                                      fileroots=self.fileroots,
                                      fileroot_noseg=self.fileroot_noseg,
                                      occultation_type=self.occultation_type,
                                      max_iter=max_iter, thresh=thresh,
                                      box_size=box_size)
            results, deepframe = step_results

        return results, deepframe


class TracingStep:
    """Wrapper around custom Tracing Step.
    """

    def __init__(self, input_data, deepframe, output_dir='./'):
        """Step initializer.
        """

        self.output_dir = output_dir
        self.deepframe = deepframe
        self.datafiles = utils.sort_datamodels(input_data)
        self.fileroots = utils.get_filename_root(self.datafiles)
        self.fileroot_noseg = utils.get_filename_root_noseg(self.fileroots)

    def run(self, generate_tracemask=True, mask_width=45, pixel_flags=None,
            calculate_stability=True, stability_params='ALL', nthreads=4,
            save_results=True, force_redo=False, generate_lc=True,
            baseline_ints=None, occultation_type='transit',
            smoothing_scale=None):
        """Method to run the step.
        """

        all_files = glob.glob(self.output_dir + '*')
        # If an output file for this segment already exists, skip the step.
        suffix = 'centroids.csv'
        expected_file = self.output_dir + self.fileroot_noseg + suffix
        if expected_file in all_files and force_redo is False:
            fancyprint('Main output file already exists.')
            fancyprint('If you wish to still produce secondary outputs, run '
                       'with force_redo=True.\n')
            fancyprint('Skipping Tracing Step.\n')
            centroids = pd.read_csv(expected_file, comment='#')
            tracemask, smoothed_lc = None, None
        # If no output files are detected, run the step.
        else:
            step_results = tracingstep(self.datafiles, self.deepframe,
                                       calculate_stability=calculate_stability,
                                       stability_params=stability_params,
                                       nthreads=nthreads,
                                       generate_tracemask=generate_tracemask,
                                       mask_width=mask_width,
                                       pixel_flags=pixel_flags,
                                       generate_lc=generate_lc,
                                       baseline_ints=baseline_ints,
                                       occultation_type=occultation_type,
                                       smoothing_scale=smoothing_scale,
                                       output_dir=self.output_dir,
                                       save_results=save_results,
                                       fileroot_noseg=self.fileroot_noseg)
            centroids, tracemask, smoothed_lc = step_results

        return centroids, tracemask, smoothed_lc


def backgroundstep(datafiles, background_model, output_dir='./',
                   save_results=True, fileroots=None, fileroot_noseg='',
                   scale1=None, scale2=None):
    """Background subtraction must be carefully treated with SOSS observations.
    Due to the extent of the PSF wings, there are very few, if any,
    non-illuminated pixels to serve as a sky region. Furthermore, the zodi
    background has a unique stepped shape, which would render a constant
    background subtraction ill-advised. Therefore, a background subtracton is
    performed by scaling a model background to the countns level of a median
    stack of the exposure. This scaled model background is then subtracted
    from each integration.

    Parameters
    ----------
    datafiles : array-like[str], array-like[CubeModel]
        Paths to data segments for a SOSS exposure, or the datamodels
        themselves.
    background_model : array-like[float]
        Background model. Should be 2D (dimy, dimx)
    output_dir : str
        Directory to which to save outputs.
    save_results : bool
        If True, save outputs to file.
    fileroots : array-like[str]
        Root names for output files.
    fileroot_noseg : str
        Root name with no segment information.
    scale1 : float, None
        Scaling value to apply to background model to match data. Will take
        precedence over calculated scaling value. If only scale1 is provided,
        this will multiply the entire frame. If scale2 is also provided, this
        will be the "pre-stp" scaling.
    scale2 : float, None
        "Post-step" scaling value. scale1 must also be passed if this
        parameter is not None.

    Returns
    -------
    results : array-like[CubeModel]
        Input data segments, corrected for the background.
    model_scaled : array-like[float]
        Background model, scaled to the flux level of each group median.
    """

    fancyprint('Starting background subtraction step.')
    # Output directory formatting.
    if output_dir is not None:
        if output_dir[-1] != '/':
            output_dir += '/'

    datafiles = np.atleast_1d(datafiles)
    opened_datafiles = []
    # Load in each of the datafiles.
    for i, file in enumerate(datafiles):
        currentfile = utils.open_filetype(file)
        opened_datafiles.append(currentfile)
        # To create the deepstack, join all segments together.
        if i == 0:
            cube = currentfile.data
        else:
            cube = np.concatenate([cube, currentfile.data], axis=0)
    datafiles = opened_datafiles

    # Make median stack of all integrations to use for background scaling.
    # This is to limit the influence of cosmic rays, which can greatly effect
    # the background scaling factor calculated for an individual inegration.
    fancyprint('Generating a deep stack using all integrations.')
    deepstack = utils.make_deepstack(cube)
    # If applied at the integration level, reshape deepstack to 3D.
    if np.ndim(deepstack) != 3:
        dimy, dimx = np.shape(deepstack)
        deepstack = deepstack.reshape(1, dimy, dimx)
    ngroup, dimy, dimx = np.shape(deepstack)

    fancyprint('Calculating background model scaling.')
    model_scaled = np.zeros_like(deepstack)
    if scale1 is None:
        fancyprint(' Scale factor(s):')
    else:
        fancyprint(' Using user-defined background scaling(s):')
        if scale2 is not None:
            fancyprint('  Pre-step scale factor: {:.5f}'.format(scale1))
            fancyprint('  Post-step scale factor: {:.5f}'.format(scale2))
        else:
            fancyprint('  Background scale factor: {:.5f}'.format(scale1))
    first_time = True
    for i in range(ngroup):
        if scale1 is None:
            # Calculate the scaling of the model background to the median
            # stack.
            if dimy == 96:
                # Use area in bottom left corner of detector for SUBSTRIP96.
                xl, xu = 5, 21
                yl, yu = 5, 401
            else:
                # Use area in the top left corner of detector for SUBSTRIP256
                xl, xu = 210, 250
                yl, yu = 250, 500
            bkg_ratio = deepstack[i, xl:xu, yl:yu] / background_model[xl:xu, yl:yu]
            # Instead of a straight median, use the median of the 2nd quartile
            # to limit the effect of any remaining illuminated pixels.
            q1 = np.nanpercentile(bkg_ratio, 25)
            q2 = np.nanpercentile(bkg_ratio, 50)
            ii = np.where((bkg_ratio > q1) & (bkg_ratio < q2))
            scale_factor = np.nanmedian(bkg_ratio[ii])
            if scale_factor < 0:
                scale_factor = 0
            fancyprint('  Background scale factor: {1:.5f}'.format(i + 1, scale_factor))
            model_scaled[i] = background_model * scale_factor
        elif scale1 is not None and scale2 is None:
            # If using a user specified scaling for the whole frame.
            model_scaled[i] = background_model * scale1
        else:
            # If using seperate pre- and post- step scalings.
            # Locate the step position using the gradient of the background.
            grad_bkg = np.gradient(background_model, axis=1)
            step_pos = np.argmax(grad_bkg[:, 10:-10], axis=1)
            # Seperately scale both sides of the step.
            for j in range(dimy):
                model_scaled[i, j, :(step_pos[j]+8)] = background_model[j, :(step_pos[j]+8)] * scale1
                model_scaled[i, j, (step_pos[j]+8):] = background_model[j, (step_pos[j]+8):] * scale2

    # Loop over all segments in the exposure and subtract the background from
    # each of them.
    results = []
    for i, currentfile in enumerate(datafiles):
        # Subtract the scaled background model.
        data_backsub = currentfile.data - model_scaled
        currentfile.data = data_backsub

        # Save the results to file if requested.
        if save_results is True:
            if first_time is True:
                # Scaled model background.
                np.save(output_dir + fileroot_noseg + 'background.npy',
                        model_scaled)
                first_time = False
            # Background subtracted data.
            currentfile.write(output_dir + fileroots[i] + 'backgroundstep.fits')

        results.append(currentfile)
        currentfile.close()

    return results, model_scaled


def badpixstep(datafiles, baseline_ints, smoothed_wlc=None, thresh=10,
               box_size=5, max_iter=1, output_dir='./', save_results=True,
               fileroots=None, fileroot_noseg='', occultation_type='transit'):
    """Identify and correct hot pixels remaining in the dataset. Find outlier
    pixels in the median stack and correct them via the median of a box of
    surrounding pixels. Then replace these pixels in each integration via the
    wlc scaled median.

    Parameters
    ----------
    datafiles : array-like[str], array-like[RampModel]
        List of paths to datafiles for each segment, or the datamodels
        themselves.
    baseline_ints : array-like[int]
        Integrations of ingress and egress.
    smoothed_wlc : array-like[float]
        Estimate of the normalized light curve.
    thresh : int
        Sigma threshold for a deviant pixel to be flagged.
    box_size : int
        Size of box around each pixel to test for deviations.
    max_iter : int
        Maximum number of outlier flagging iterations.
    output_dir : str
        Directory to which to output results.
    save_results : bool
        If True, save results to file.
    fileroots : array-like[str], None
        Root names for output files.
    fileroot_noseg : str
        Root file name with no segment information.
    occultation_type : str
        Type of occultation, either 'transit' or 'eclipse'.

    Returns
    -------
    data : list[CubeModel]
        Input datamodels for each segment, corrected for outlier pixels.
    deepframe : array-like[float]
        Final median stack of all outlier corrected integrations.
    """

    fancyprint('Starting hot pixel interpolation step.')

    # Output directory formatting.
    if output_dir is not None:
        if output_dir[-1] != '/':
            output_dir += '/'

    datafiles = np.atleast_1d(datafiles)
    # Format the baseline frames - either out-of-transit or in-eclipse.
    baseline_ints = utils.format_out_frames(baseline_ints,
                                            occultation_type)

    data = []
    # Load in datamodels from all segments.
    for i, file in enumerate(datafiles):
        currentfile = utils.open_filetype(file)
        data.append(currentfile)

        # To create the deepstack, join all segments together.
        # Also stack all the dq arrays from each segement.
        if i == 0:
            cube = currentfile.data
            dq_cube = currentfile.dq
        else:
            cube = np.concatenate([cube, currentfile.data], axis=0)
            dq_cube = np.concatenate([dq_cube, currentfile.dq], axis=0)

    # Initialize starting loop variables.
    newdata = np.copy(cube)
    newdq = np.copy(dq_cube)
    it = 0

    while it < max_iter:
        fancyprint('Starting iteration {0} of {1}.'.format(it + 1, max_iter))

        # Generate the deepstack.
        fancyprint(' Generating a deep stack...')
        deepframe = utils.make_deepstack(newdata[baseline_ints])
        badpix = np.zeros_like(deepframe)
        count = 0
        nint, dimy, dimx = np.shape(newdata)

        # Loop over whole deepstack and flag deviant pixels.
        for i in tqdm(range(4, dimx-4)):
            for j in range(dimy):
                box_size_i = box_size
                box_prop = utils.get_interp_box(deepframe, box_size_i, i, j,
                                                dimx)
                # Ensure that the median and std dev extracted are good.
                # If not, increase the box size until they are.
                while np.any(np.isnan(box_prop)):
                    box_size_i += 1
                    box_prop = utils.get_interp_box(deepframe, box_size_i, i,
                                                    j, dimx)
                med, std = box_prop[0], box_prop[1]

                # If central pixel is too deviant (or nan/negative) flag it.
                if np.abs(deepframe[j, i] - med) >= (thresh * std) or np.isnan(deepframe[j, i]) or deepframe[j, i] < 0:
                    mini, maxi = np.max([0, i - 1]), np.min([dimx - 1, i + 1])
                    minj, maxj = np.max([0, j - 1]), np.min([dimy - 1, j + 1])
                    badpix[j, i] = 1
                    # Also flag cross around the central pixel.
                    badpix[maxj, i] = 1
                    badpix[minj, i] = 1
                    badpix[j, maxi] = 1
                    badpix[j, mini] = 1
                    count += 1

        fancyprint(' {} bad pixels identified this iteration.'.format(count))
        # End if no bad pixels are found.
        if count == 0:
            break
        # Replace the flagged pixels in the median integration.
        newdeep, deepdq = utils.do_replacement(deepframe, badpix,
                                               dq=np.ones_like(deepframe),
                                               box_size=box_size)

        # If no lightcurve is provided, estimate it from the current data.
        if smoothed_wlc is None:
            postage = cube[:, 20:60, 1500:1550]
            timeseries = np.nansum(postage, axis=(1, 2))
            timeseries = timeseries / np.nanmedian(timeseries[baseline_ints])
            # Smooth the time series on a timescale of roughly 2%.
            smoothed_wlc = median_filter(timeseries,
                                         int(0.02*np.shape(cube)[0]))
        # Replace hot pixels in each integration using a scaled median.
        newdeep = np.repeat(newdeep, nint).reshape(dimy, dimx, nint)
        newdeep = newdeep.transpose(2, 0, 1) * smoothed_wlc[:, None, None]
        mask = badpix.astype(bool)
        newdata[:, mask] = newdeep[:, mask]
        # Set DQ flags for these pixels to zero (use the pixel).
        deepdq = ~deepdq.astype(bool)
        newdq[:, deepdq] = 0

        it += 1
        thresh += 1

    # Generate a final corrected deep frame for the baseline integrations.
    deepframe = utils.make_deepstack(newdata[baseline_ints])

    current_int = 0
    # Save interpolated data.
    for n, file in enumerate(data):
        currentdata = file.data
        nints = np.shape(currentdata)[0]
        file.data = newdata[current_int:(current_int + nints)]
        file.dq = newdq[current_int:(current_int + nints)]
        current_int += nints
        if save_results is True:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                file.write(output_dir + fileroots[n] + 'badpixstep.fits')
        file.close()

    if save_results is True:
        # Save deep frame.
        hdu = fits.PrimaryHDU(deepframe)
        hdu.writeto(output_dir + fileroot_noseg + 'deepframe.fits',
                    overwrite=True)

    return data, deepframe


def tracingstep(datafiles, deepframe=None, calculate_stability=True,
                stability_params='ALL', nthreads=4, generate_tracemask=True,
                mask_width=45, pixel_flags=None, generate_lc=True,
                baseline_ints=None, occultation_type='transit',
                smoothing_scale=None, output_dir='./', save_results=True,
                fileroot_noseg=''):
    """A multipurpose step to perform some initial analysis of the 2D
    dataframes and produce products which can be useful in further reduction
    iterations. The four functionalities are detailed below:
    1. Locate the centroids of all three SOSS orders via the edgetrigger
    algorithm.
    2. (optional) Generate a mask of the target diffraction orders.
    3. (optional) Calculate the stability of the SOSS traces over the course
    of the TSO.
    4. (optional) Create a smoothed estimate of the order 1 white light curve.

    Parameters
    ----------
    datafiles : array-like[str], array-like[RampModel]
        List of paths to datafiles for each segment, or the datamodels
        themselves.
    deepframe : str, array-like[float], None
        Path to median stack file, or the median stack itself. Should be 2D
        (dimy, dimx). If None is passed, one will be generated.
    calculate_stability : bool
        If True, calculate the stabilty of the SOSS trace over the TSO.
    stability_params : str, array-like[str]
        List of parameters for which to calculate the stability. Any of: 'x',
        'y', and/or 'FWHM', or 'ALL' for all three.
    nthreads : int
        Number of CPUs for stability parameter calculation multiprocessing.
    generate_tracemask : bool
        If True, generate a mask of the target diffraction orders.
    mask_width : int
        Mask width, in pixels, around the trace centroids. Only necesssary if
        generate_tracemask is True.
    pixel_flags: None, str, array-like[str]
        Paths to files containing existing pixel flags to which the trace mask
        should be added. Only necesssary if generate_tracemask is True.
    generate_lc : bool
        If True, also produce a smoothed order 1 white light curve.
    baseline_ints : array-like[int]
        Integrations of ingress and egress. Only necessary if generate_lc=True.
    occultation_type : str
        Type of occultation, either 'transit' or 'eclipse'. Only necessary if
        generate_lc=True.
    smoothing_scale : int, None
        Timescale on which to smooth the lightcurve. Only necessary if
        generate_lc=True.
    output_dir : str
        Directory to which to save outputs.
    save_results : bool
        If Tre, save results to file.
    fileroot_noseg : str
        Root file name with no segment information.

    Returns
    -------
    centroids : array-like[float]
        Trace centroids for all three orders.
    tracemask : array-like[bool], None
        If requested, the trace mask.
    smoothed_lc : array-like[float], None
        If requested, the smoothed order 1 white light curve.
    """

    fancyprint('Starting Tracing Step.')

    datafiles = np.atleast_1d(datafiles)
    # If no deepframe is passed, construct one. Also generate a datacube for
    # later white light curve or stability calculations.
    if deepframe is None or generate_lc is True or calculate_stability is True:
        # Construct datacube from the data files.
        for i, file in enumerate(datafiles):
            if isinstance(file, str):
                this_data = fits.getdata(file, 1)
            else:
                this_data = file.data
            if i == 0:
                cube = this_data
            else:
                cube = np.concatenate([cube, this_data], axis=0)
        deepframe = utils.make_deepstack(cube)
    elif isinstance(deepframe, str):
        deepframe = fits.getdata(deepframe)
    # Get the subarray dimensions.
    dimy, dimx = np.shape(deepframe)
    if dimy == 256:
        subarray = 'SUBSTRIP256'
    elif dimy == 96:
        subarray = 'SUBSTRIP96'
    else:
        raise NotImplementedError

    # ===== PART 1: Get centroids for orders one to three =====
    fancyprint('Finding trace centroids.')
    # Get the most up to date trace table file.
    step = calwebb_spec2.extract_1d_step.Extract1dStep()
    tracetable = step.get_reference_file(datafiles[0], 'spectrace')
    # Get centroids via the edgetrigger method.
    save_filename = output_dir + fileroot_noseg
    centroids = utils.get_trace_centroids(deepframe, tracetable, subarray,
                                          save_results=save_results,
                                          save_filename=save_filename)
    x1, y1 = centroids[0][0], centroids[0][1]
    x2, y2 = centroids[1][0], centroids[1][1]
    x3, y3 = centroids[2][0], centroids[2][1]

    # ===== PART 2: Create trace masks for each order =====
    # If requested, create a trace mask for each order.
    tracemask = None
    if generate_tracemask is True:
        weights1 = soss_boxextract.get_box_weights(y1, mask_width,
                                                   (dimy, dimx),
                                                   cols=x1.astype(int))
        weights1 = np.where(weights1 == 0, 0, 1)
        # Make masks for other 2 orders for SUBSTRIP256.
        if subarray != 'SUBSTRIP96':
            weights2 = soss_boxextract.get_box_weights(y2, mask_width,
                                                       (dimy, dimx),
                                                       cols=x2.astype(int))
            weights2 = np.where(weights2 == 0, 0, 1)
            weights3 = soss_boxextract.get_box_weights(y3, mask_width,
                                                       (dimy, dimx),
                                                       cols=x3.astype(int))
            weights3 = np.where(weights3 == 0, 0, 1)
        else:
            weights2 = np.zeros_like(weights1)
            weights3 = np.zeros_like(weights1)

        # Combine the masks for each order.
        tracemask = weights1.astype(bool) | weights2.astype(bool) | \
            weights3.astype(bool)

        # Save the trace mask to file if requested.
        if save_results is True:
            # If we are to combine the trace mask with existing pixel mask.
            if pixel_flags is not None:
                pixel_flags = np.atleast_1d(pixel_flags)
                # Ensure there is one pixel flag file per data file
                assert len(pixel_flags) == len(datafiles)
                # Combine tracemask with existing flags and overwrite old file.
                for flag_file in pixel_flags:
                    old_flags = fits.getdata(flag_file)
                    new_flags = old_flags.astype(bool) | tracemask
                    hdu = fits.PrimaryHDU(new_flags.astype(int))
                    hdu.writeto(flag_file, overwrite=True)
            else:
                hdu = fits.PrimaryHDU(tracemask.astype(int))
                suffix = 'tracemask_width{}.fits'.format(mask_width)
                outfile = output_dir + fileroot_noseg + suffix
                hdu.writeto(outfile, overwrite=True)
                fancyprint('Trace mask saved to {}'.format(outfile))

    # ===== PART 3: Calculate the trace stability =====
    # If requested, calculate the change in position of the trace, as well as
    # its FWHM over the course of the TSO. These quantities may be useful for
    # lightcurve detrending.
    if calculate_stability is True:
        fancyprint('Calculating trace stability... This might take a while.')
        assert save_results is True, 'save_results must be True to run ' \
                                     'soss_stability'
        if stability_params == 'ALL':
            stability_params = ['x', 'y', 'FWHM']

        # Calculate the stability of the requested parameters.
        stability_results = {}
        if 'x' in stability_params:
            fancyprint('Getting trace X-positions...')
            ccf_x = utils.soss_stability(cube, axis='x', nthreads=nthreads)
            stability_results['X'] = ccf_x
        if 'y' in stability_params:
            fancyprint('Getting trace Y-positions...')
            ccf_y = utils.soss_stability(cube, axis='y', nthreads=nthreads)
            stability_results['Y'] = ccf_y
        if 'FWHM' in stability_params:
            fancyprint('Getting trace FWHM values...')
            fwhm = utils.soss_stability_fwhm(cube, y1, nthreads=nthreads)
            stability_results['FWHM'] = fwhm

        # Save stability results.
        df = pd.DataFrame(data=stability_results)
        suffix = 'soss_stability.csv'
        if os.path.exists(output_dir + fileroot_noseg + suffix):
            os.remove(output_dir + fileroot_noseg + suffix)
        df.to_csv(output_dir + fileroot_noseg + suffix, index=False)

    # ===== PART 4: Calculate a smoothed light curve =====
    # If requested, generate a smoothed estimate of the order 1 white light
    # curve.
    smoothed_lc = None
    if generate_lc is True:
        # Format the baseline frames - either out-of-transit or in-eclipse.
        assert baseline_ints is not None
        baseline_ints = utils.format_out_frames(baseline_ints,
                                                occultation_type)

        # Use an area centered on the peak of the order 1 blaze to estimate the
        # photometric light curve.
        postage = cube[:, 20:60, 1500:1550]
        timeseries = np.nansum(postage, axis=(1, 2))
        # Normalize by the baseline flux level.
        timeseries = timeseries / np.nanmedian(timeseries[baseline_ints])
        # If not smoothing scale is provided, smooth the time series on a
        # timescale of roughly 2% of the total length.
        if smoothing_scale is None:
            smoothing_scale = int(0.02 * np.shape(cube)[0])
        smoothed_lc = median_filter(timeseries, smoothing_scale)

        if save_results is True:
            outfile = output_dir + fileroot_noseg + 'lcestimate.npy'
            fancyprint('Smoothed light curve saved to {}'.format(outfile))
            np.save(outfile, smoothed_lc)

    return centroids, tracemask, smoothed_lc


def run_stage2(results, background_model, baseline_ints, smoothed_wlc=None,
               save_results=True, force_redo=False, calculate_stability=True,
               stability_params='ALL', nthreads=4, root_dir='./',
               output_tag='', occultation_type='transit', smoothing_scale=None,
               skip_steps=None, generate_lc=True, generate_tracemask=True,
               mask_width=45, pixel_flags=None, **kwargs):
    """Run the supreme-SPOON Stage 2 pipeline: spectroscopic processing,
    using a combination of official STScI DMS and custom steps. Documentation
    for the official DMS steps can be found here:
    https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_spec2.html

    Parameters
    ----------
    results : array-like[str], array-like[CubeModel]
        supreme-SPOON Stage 1 output files.
    background_model : array-like[float]
        SOSS background model.
    baseline_ints : array-like[int]
        Integrations of ingress and egress.
    smoothed_wlc : array-like[float], None
        Estimate of the normalized light curve.
    save_results : bool
        If True, save results of each step to file.
    force_redo : bool
        If True, redo steps even if outputs files are already present.
    calculate_stability : bool
        If True, calculate the stability of the SOSS trace over the course of
        the TSO.
    stability_params : str, array-like[str]
        List of parameters for which to calculate the stability. Any of: 'x',
        'y', and/or 'FWHM', or 'ALL' for all three.
    nthreads : int
        Number of CPUs for stability parameter calculation multiprocessing.
    root_dir : str
        Directory from which all relative paths are defined.
    output_tag : str
        Name tag to append to pipeline outputs directory.
    occultation_type : str
        Type of occultation: transit or eclipse.
    smoothing_scale : int, None
        Timescale on which to smooth the lightcurve.
    skip_steps : array-like[str], None
        Step names to skip (if any).
    generate_lc : bool
        If True, produce a smoothed order 1 white light curve.
    generate_tracemask : bool
        If True, generate a mask of the target diffraction orders.
    mask_width : int
        Mask width, in pixels, around the trace centroids. Only necesssary if
        generate_tracemask is True.
    pixel_flags: None, str, array-like[str]
        Paths to files containing existing pixel flags to which the trace mask
        should be added. Only necesssary if generate_tracemask is True.

    Returns
    -------
    results : array-like[CubeModel]
        Datafiles for each segment processed through Stage 2.
    deepframe : array-like[float]
        Median stack of the baseline flux level integrations (i.e.,
        out-of-transit or in-eclipse).
    """

    # ============== DMS Stage 2 ==============
    # Spectroscopic processing.
    fancyprint('\n\n**Starting supreme-SPOON Stage 2**')
    fancyprint('Spectroscopic processing\n\n')

    if output_tag != '':
        output_tag = '_' + output_tag
    # Create output directories and define output paths.
    utils.verify_path(root_dir + 'pipeline_outputs_directory' + output_tag)
    utils.verify_path(root_dir + 'pipeline_outputs_directory' + output_tag + '/Stage2')
    outdir = root_dir + 'pipeline_outputs_directory' + output_tag + '/Stage2/'

    if skip_steps is None:
        skip_steps = []

    # ===== Assign WCS Step =====
    # Default DMS step.
    if 'AssignWCSStep' not in skip_steps:
        if 'AssignWCSStep' in kwargs.keys():
            step_kwargs = kwargs['AssignWCSStep']
        else:
            step_kwargs = {}
        step = AssignWCSStep(results, output_dir=outdir)
        results = step.run(save_results=save_results, force_redo=force_redo,
                           **step_kwargs)

    # ===== Source Type Determination Step =====
    # Default DMS step.
    if 'SourceTypeStep' not in skip_steps:
        if 'SourceTypeStep' in kwargs.keys():
            step_kwargs = kwargs['SourceTypeStep']
        else:
            step_kwargs = {}
        step = SourceTypeStep(results, output_dir=outdir)
        results = step.run(save_results=save_results, force_redo=force_redo,
                           **step_kwargs)

    # ===== Background Subtraction Step =====
    # Custom DMS step.
    if 'BackgroundStep' not in skip_steps:
        if 'BackgroundStep' in kwargs.keys():
            step_kwargs = kwargs['BackgroundStep']
        else:
            step_kwargs = {}
        step = BackgroundStep(results, background_model=background_model,
                              output_dir=outdir)
        results = step.run(save_results=save_results, force_redo=force_redo,
                           **step_kwargs)[0]

    # ===== Flat Field Correction Step =====
    # Default DMS step.
    if 'FlatFieldStep' not in skip_steps:
        if 'FlatFieldStep' in kwargs.keys():
            step_kwargs = kwargs['FlatFieldStep']
        else:
            step_kwargs = {}
        step = FlatFieldStep(results, output_dir=outdir)
        results = step.run(save_results=save_results, force_redo=force_redo,
                           **step_kwargs)

    # ===== Hot Pixel Correction Step =====
    # Custom DMS step.
    if 'BadPixStep' not in skip_steps:
        step = BadPixStep(results, baseline_ints=baseline_ints,
                          smoothed_wlc=smoothed_wlc, output_dir=outdir,
                          occultation_type=occultation_type)
        step_results = step.run(save_results=save_results,
                                force_redo=force_redo)
        results, deepframe = step_results
    else:
        deepframe = None

    # ===== Tracing Step =====
    # Custom DMS step.
    if 'TracingStep' not in skip_steps:
        step = TracingStep(results, deepframe=deepframe, output_dir=outdir)
        centroids = step.run(calculate_stability=calculate_stability,
                             stability_params=stability_params,
                             nthreads=nthreads,
                             generate_tracemask=generate_tracemask,
                             mask_width=mask_width, pixel_flags=pixel_flags,
                             generate_lc=generate_lc,
                             baseline_ints=baseline_ints,
                             occultation_type=occultation_type,
                             smoothing_scale=smoothing_scale,
                             save_results=save_results)

    return results, deepframe
