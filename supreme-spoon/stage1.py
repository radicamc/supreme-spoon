#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Jul 21 17:30 2022

@author: MCR

Custom JWST DMS pipeline steps for Stage 1 (detector level processing).
"""

from astropy.io import fits
import glob
import numpy as np
from tqdm import tqdm
import warnings

from jwst import datamodels
from jwst.pipeline import calwebb_detector1

from supreme_spoon import utils, plotting
from supreme_spoon.databowl import DataBowl


class GroupScaleStep:
    """Wrapper around default calwebb_detector1 Group Scale Correction step.
    """
    def __init__(self, input_data, output_dir='./'):
        self.tag = 'groupscalestep.fits'
        self.output_dir = output_dir
        if isinstance(input_data, DataBowl):
            self.datafiles = input_data.datamodels
            self.fileroots = input_data.fileroots
        else:
            self.datafiles = np.atleast_1d(input_data)
            self.fileroots = utils.get_filename_root(self.datafiles)

    def run(self, save_results=True, force_redo=False, **kwargs):
        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                print('Output file {} already exists.'.format(expected_file))
                print('Skipping Group Scale Step.\n')
                res = expected_file
            # If no output files are detected, run the step.
            else:
                step = calwebb_detector1.group_scale_step.GroupScaleStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results, **kwargs)
            results.append(res)

        return results


class DQInitStep:
    """Wrapper around default calwebb_detector1 Data Quality Initialization
    step.
    """

    def __init__(self, input_data, output_dir='./'):
        self.tag = 'dqnitstep.fits'
        self.output_dir = output_dir
        if isinstance(input_data, DataBowl):
            self.datafiles = input_data.datamodels
            self.fileroots = input_data.fileroots
        else:
            self.datafiles = np.atleast_1d(input_data)
            self.fileroots = utils.get_filename_root(self.datafiles)

    def run(self, save_results=True, force_redo=False, **kwargs):
        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                print('Output file {} already exists.'.format(expected_file))
                print('Skipping Data Quality Initialization Step.\n')
                res = expected_file
            # If no output files are detected, run the step.
            else:
                step = calwebb_detector1.dq_init_step.DQInitStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results, **kwargs)
            results.append(res)

        return results


class SaturationStep:
    """Wrapper around default calwebb_detector1 Saturation Detection step.
    """

    def __init__(self, input_data, output_dir='./'):
        self.tag = 'saturationstep.fits'
        self.output_dir = output_dir
        if isinstance(input_data, DataBowl):
            self.datafiles = input_data.datamodels
            self.fileroots = input_data.fileroots
        else:
            self.datafiles = np.atleast_1d(input_data)
            self.fileroots = utils.get_filename_root(self.datafiles)

    def run(self, save_results=True, force_redo=False, **kwargs):
        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                print('Output file {} already exists.'.format(expected_file))
                print('Skipping Saturation Detection Step.\n')
                res = expected_file
            # If no output files are detected, run the step.
            else:
                step = calwebb_detector1.saturation_step.SaturationStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results, **kwargs)
            results.append(res)

        return results


class SuperBiasStep:
    """Wrapper around default calwebb_detector1 Super Bias Subtraction step.
    """

    def __init__(self, input_data, output_dir='./'):
        self.tag = 'superbiasstep.fits'
        self.output_dir = output_dir
        if isinstance(input_data, DataBowl):
            self.datafiles = input_data.datamodels
            self.fileroots = input_data.fileroots
        else:
            self.datafiles = np.atleast_1d(input_data)
            self.fileroots = utils.get_filename_root(self.datafiles)

    def run(self, save_results=True, force_redo=False, **kwargs):
        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                print('Output file {} already exists.'.format(expected_file))
                print('Skipping Superbias Subtraction Step.\n')
                res = expected_file
            # If no output files are detected, run the step.
            else:
                step = calwebb_detector1.superbias_step.SuperBiasStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results, **kwargs)
            results.append(res)

        return results


class RefPixStep:
    """Wrapper around default calwebb_detector1 Reference Pixel Correction
    step.
    """

    def __init__(self, input_data, output_dir='./'):
        self.tag = 'refpixstep.fits'
        self.output_dir = output_dir
        if isinstance(input_data, DataBowl):
            self.datafiles = input_data.datamodels
            self.fileroots = input_data.fileroots
        else:
            self.datafiles = np.atleast_1d(input_data)
            self.fileroots = utils.get_filename_root(self.datafiles)

    def run(self, save_results=True, force_redo=False, **kwargs):
        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                print('Output file {} already exists.'.format(expected_file))
                print('Skipping Reference Pixel Correction Step.\n')
                res = expected_file
            # If no output files are detected, run the step.
            else:
                step = calwebb_detector1.refpix_step.RefPixStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results, **kwargs)
            results.append(res)

        return results


class BackgroundStep:
    """Wrapper around custom Background Subtraction step.
    """
    def __init__(self, input_data, background_model, output_dir='./'):
        self.tag = 'backgroundstep.fits'
        self.output_dir = output_dir
        if isinstance(input_data, DataBowl):
            self.datafiles = input_data.datamodels
            self.fileroots = input_data.fileroots
        else:
            self.datafiles = np.atleast_1d(input_data)
            self.fileroots = utils.get_filename_root(self.datafiles)
        self.background_model = np.load(background_model)

    def run(self, save_results=True, force_redo=False):
        all_files = glob.glob(self.output_dir + '*')
        do_step = 1
        results, background_models = [], []
        for i in range(len(self.datafiles)):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            expected_bkg = self.output_dir + self.fileroots[i] + 'background.fits'
            if expected_file not in all_files:
                do_step *= 0
            else:
                results.append(datamodels.open(expected_file))
                background_models.append(fits.getdata(expected_bkg))
        if do_step == 1 and force_redo is False:
            print('Output files already exist.')
            print('Skipping Background Subtraction Step.')
        # If no output files are detected, run the step.
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                step_results = backgroundstep(self.datafiles,
                                              self.background_model,
                                              output_dir=self.output_dir,
                                              save_results=save_results,
                                              fileroots=self.fileroots)
                results, background_models = step_results

        return results, background_models


class OneOverFStep:
    """Wrapper around custom 1/f Correction Step.
    """
    def __init__(self, input_data, baseline_ints, output_dir='./',
                 smoothed_wlc=None, outlier_maps=None, trace_mask=None):
        self.tag = 'oneoverfstep.fits'
        self.output_dir = output_dir
        self.baseline_ints = baseline_ints
        self.smoothed_wlc = smoothed_wlc
        self.trace_mask = trace_mask
        self.outlier_maps = outlier_maps

        if isinstance(input_data, DataBowl):
            self.datafiles = input_data.datamodels
            self.fileroots = input_data.fileroots
            if input_data.smoothed_lightcurve is not None:
                self.scaling_curve = input_data.smoothed_lightcurve
            if input_data.trace_mask is not None:
                self.trace_mask = input_data.trace_mask
            if input_data.baseline_ints is not None:
                self.baseline_ints = input_data.baseline_ints
            if input_data.outlier_maps is not None:
                self.outlier_maps = input_data.outlier_maps
        else:
            self.datafiles = np.atleast_1d(input_data)
            self.fileroots = utils.get_filename_root(self.datafiles)

    def run(self, save_results=True, force_redo=False):
        all_files = glob.glob(self.output_dir + '*')
        do_step = 1
        results = []
        for i in range(len(self.datafiles)):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file not in all_files:
                do_step *= 0
            else:
                results.append(datamodels.open(expected_file))
        if do_step == 1 and force_redo is False:
            print('Output files already exist.')
            print('Skipping 1/f Correction Step.\n')
        # If no output files are detected, run the step.
        else:
            results = oneoverfstep(self.datafiles,
                                   baseline_ints=self.baseline_ints,
                                   smoothed_wlc=self.smoothed_wlc,
                                   output_dir=self.output_dir,
                                   save_results=save_results,
                                   outlier_maps=self.outlier_maps,
                                   trace_mask=self.trace_mask,
                                   fileroots=self.fileroots)

        return results


class LinearityStep:
    """Wrapper around default calwebb_detector1 Linearity Correction step.
    """

    def __init__(self, input_data, output_dir='./'):
        self.tag = 'linearitystep.fits'
        self.output_dir = output_dir
        if isinstance(input_data, DataBowl):
            self.datafiles = input_data.datamodels
            self.fileroots = input_data.fileroots
        else:
            self.datafiles = np.atleast_1d(input_data)
            self.fileroots = utils.get_filename_root(self.datafiles)

    def run(self, save_results=True, force_redo=False, **kwargs):
        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                print('Output file {} already exists.'.format(expected_file))
                print('Skipping Linearity Correction Step.\n')
                res = expected_file
            # If no output files are detected, run the step.
            else:
                step = calwebb_detector1.linearity_step.LinearityStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results, **kwargs)
                # Hack to remove oneoverfstep tag from file name.
                try:
                    res = utils.fix_filenames(res, '_oneoverfstep_',
                                              self.output_dir)[0]
                    res = datamodels.open(res)
                except IndexError:
                    pass
            results.append(res)

        return results


class JumpStep:
    """Wrapper around default calwebb_detector1 Jump Detection step.
    """

    def __init__(self, input_data, output_dir='./'):
        self.tag = 'jump.fits'
        self.output_dir = output_dir
        if isinstance(input_data, DataBowl):
            self.datafiles = input_data.datamodels
            self.fileroots = input_data.fileroots
        else:
            self.datafiles = np.atleast_1d(input_data)
            self.fileroots = utils.get_filename_root(self.datafiles)

    def run(self, save_results=True, force_redo=False, rejection_threshold=5,
            **kwargs):
        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                print('Output file {} already exists.'.format(expected_file))
                print('Skipping Jump Detection Step.\n')
                res = expected_file
            # If no output files are detected, run the step.
            else:
                step = calwebb_detector1.jump_step.JumpStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results,
                                rejection_threshold=rejection_threshold,
                                maximum_cores='quarter', **kwargs)
            results.append(res)

        return results


class RampFitStep:
    """Wrapper around default calwebb_detector1 Ramp Fit step.
    """
    def __init__(self, input_data, output_dir='./'):
        self.tag = 'jump.fits'
        self.output_dir = output_dir
        if isinstance(input_data, DataBowl):
            self.datafiles = input_data.datamodels
            self.fileroots = input_data.fileroots
        else:
            self.datafiles = np.atleast_1d(input_data)
            self.fileroots = utils.get_filename_root(self.datafiles)

    def run(self, save_results=True, force_redo=False, **kwargs):
        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                print('Output file {} already exists.'.format(expected_file))
                print('Skipping Ramp Fit Step.\n')
                res = expected_file
            # If no output files are detected, run the step.
            else:
                step = calwebb_detector1.ramp_fit_step.RampFitStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results, **kwargs)[1]
                # Store pixel flags in seperate files for potential use in 1/f
                # noise correction.
                hdu = fits.PrimaryHDU(res.dq)
                outfile = self.output_dir + self.fileroots[i] + 'dqpixelflags.fits'
                hdu.writeto(outfile, overwrite=True)
                # Hack to remove _1_ tag from file name.
                res = utils.fix_filenames(res, '_1_', self.output_dir)[0]
            results.append(res)

        return results


class GainScaleStep:
    """Wrapper around default calwebb_detector1 Gain Scale Correction step.
    """

    def __init__(self, input_data, output_dir='./'):
        self.tag = 'gainscalestep.fits'
        self.output_dir = output_dir
        if isinstance(input_data, DataBowl):
            self.datafiles = input_data.datamodels
            self.fileroots = input_data.fileroots
        else:
            self.datafiles = np.atleast_1d(input_data)
            self.fileroots = utils.get_filename_root(self.datafiles)

    def run(self, save_results=True, force_redo=False, **kwargs):
        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                print('Output file {} already exists.'.format(expected_file))
                print('Skipping Gain Scale Correction Step.\n')
                res = expected_file
            # If no output files are detected, run the step.
            else:
                step = calwebb_detector1.gain_scale_step.GainScaleStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results, **kwargs)
            results.append(res)

        return results


def backgroundstep(datafiles, background_model, output_dir=None,
                   save_results=True, show_plots=False, fileroots=None):
    """Background subtraction must be carefully treated with SOSS observations.
    Due to the extent of the PSF wings, there are very few, if any,
    non-illuminated pixels to serve as a sky region. Furthermore, the zodi
    background has a unique stepped shape, which would render a constant
    background subtraction ill-advised. Therefore, a background subtracton is
    performed by scaling a model background to the counntns level of a median
    stack of the exposure. This scaled model background is then subtracted
    from each integration.

    Parameters
    ----------
    datafiles : list[str], list[CubeModel], np.ndarray[str]
        Paths to data segments for a SOSS exposure, or the datamodels
        themselves.
    background_model : np.array
        Background model. Should be 2D (dimy, dimx)
    output_dir : str, None
        Directory to which to save outputs.
    save_results : bool
        If True, save outputs to file.
    show_plots : bool
        If True, show plots.
    fileroots : list[str]
        Root names for output files.

    Returns
    -------
    results : list[CubeModel]
        Input data segments, corrected for the background.
    model_scaled : np.array
        Background model, scaled to the flux level of each group median.
    """

    print('Starting background subtraction step.')
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
    print('Generating a deep stack using all integrations.')
    deepstack = utils.make_deepstack(cube)
    ngroup, dimy, dimx = np.shape(deepstack)

    model_scaled = np.zeros_like(deepstack)
    for i in range(ngroup):
        # Ccalculate the scaling of the model background to the median stack.
        if dimy == 96:
            # Use area in bottom left corner of detector for SUBSTRIP96.
            xl, xu = 10, 20
            yl, yu = 10, 200
        else:
            # Use area in the top left corner of detector for SUBSTRIP256
            xl, xu = 210, 250
            yl, yu = 500, 800
        bkg_ratio = deepstack[i, xl:xu, yl:yu] / background_model[xl:xu, yl:yu]
        # Instead of a straight median, use the median of the 2nd quartile to
        # limit the effect of any remaining illuminated pixels.
        q1 = np.nanpercentile(bkg_ratio, 25)
        q2 = np.nanpercentile(bkg_ratio, 50)
        ii = np.where((bkg_ratio > q1) & (bkg_ratio < q2))
        scale_factor = np.nanmedian(bkg_ratio[ii])
        model_scaled[i] = background_model * scale_factor

    # Loop over all segments in the exposure and subtract the background from
    # each of them.
    results = []
    for i, currentfile in enumerate(datafiles):
        # Subtract the scaled background model.
        data_backsub = currentfile.data - model_scaled
        currentfile.data = data_backsub

        # Save the results to file if requested.
        if save_results is True:
            # Scaled model background.
            hdu = fits.PrimaryHDU(model_scaled)
            hdu.writeto(output_dir + fileroots[i] + 'background.fits',
                        overwrite=True)
            # Background subtracted data.
            currentfile.write(output_dir + fileroots[i] + 'backgroundstep.fits')

        # Show background scaling plot if requested.
        if show_plots is True:
            plotting.do_backgroundsubtraction_plot(currentfile.data[:, -1],
                                                   background_model,
                                                   scale_factor)
        results.append(currentfile)
        currentfile.close()

    return results, model_scaled


def oneoverfstep(datafiles, baseline_ints, smoothed_wlc=None,
                 output_dir=None, save_results=True, outlier_maps=None,
                 trace_mask=None, use_dq=True, fileroots=None):
    """Custom 1/f correction routine to be applied at the group level. A
    median stack is constructed using all out-of-transit integrations and
    subtracted from each individual integration. The column-wise median of
    this difference image is then subtracted from the original frame to
    correct 1/f noise. Outlier pixels, as well as the trace itself can be
    masked to improve the noise level estimation.

    Parameters
    ----------
    datafiles : list[str], or list[RampModel], np.ndarray[str]
        List of paths to data files, or RampModels themselves for each segment
        of the TSO. Should be 4D ramps and not rate files.
    baseline_ints : list[int]
        Integration numbers of ingress and egress.
    smoothed_wlc : None, np.array
        Estimate of the out-of-transit normalized light curve.
    output_dir : str, None
        Directory to which to save results.
    save_results : bool
        If True, save results to disk.
    outlier_maps : list[str], None
        List of paths to outlier maps for each data segment. Can be
        3D (nints, dimy, dimx), or 2D (dimy, dimx) files.
    trace_mask : str, None
        Path to file containing a trace mask. Should be 3D (norder, dimy,
        dimx), or 2D (dimy, dimx).
    use_dq : bool
        If True, mask all pixels currently flagged in the DQ array.
    fileroots : list[str]
        Root names for output files.

    Returns
    -------
    corrected_rampmodels : list
        RampModels for each segment, corrected for 1/f noise.
    """

    print('Starting 1/f correction step.')

    # Output directory formatting.
    if output_dir is not None:
        if output_dir[-1] != '/':
            output_dir += '/'

    # Format the out of transit frames.
    baseline_ints = np.abs(baseline_ints)
    out_trans = np.concatenate([np.arange(baseline_ints[0]),
                                np.arange(baseline_ints[1]) - baseline_ints[1]])

    datafiles = np.atleast_1d(datafiles)
    # If outlier maps are passed, ensure that there is one for each segment.
    if outlier_maps is not None:
        outlier_maps = np.atleast_1d(outlier_maps)
        if len(outlier_maps) == 1:
            outlier_maps = [outlier_maps[0] for d in datafiles]

    data = []
    # Load in datamodels from all segments.
    for i, file in enumerate(datafiles):
        currentfile = utils.open_filetype(file)
        data.append(currentfile)
        # To create the deepstack, join all segments together.
        if i == 0:
            cube = currentfile.data
        else:
            cube = np.concatenate([cube, currentfile.data], axis=0)

    # Generate the 3D deep stack (ngroup, dimy, dimx) using only
    # out-of-transit integrations.
    msg = 'Generating a deep stack for each frame using out-of-transit' \
          ' integrations...'
    print(msg)
    deepstack = utils.make_deepstack(cube[out_trans])

    # In order to subtract off the trace as completely as possible, the median
    # stack must be scaled, via the transit curve, to the flux level of each
    # integration.
    # If no lightcurve is provided, estimate it from the current data.
    # TODO: smooth wlc
    if smoothed_wlc is None:
        postage = cube[:, -1, 20:60, 1500:1550]
        timeseries = np.sum(postage, axis=(1, 2))
        smoothed_wlc = timeseries / np.median(timeseries[out_trans])

    # Individually treat each segment.
    corrected_rampmodels = []
    current_int = 0
    for n, datamodel in enumerate(data):
        print('Starting segment {} of {}.'.format(n + 1, len(data)))

        # Define the readout setup.
        nint, ngroup, dimy, dimx = np.shape(datamodel.data)
        # get data quality flags if requested.
        if use_dq is True:
            print(' Considering data quality flags.')
            dq = datamodel.groupdq
            # Mask will be applied multiplicatively.
            dq = np.where(dq == 0, 1, np.nan)
        else:
            dq = np.ones_like(datamodel.data)

        # Read in the outlier map -- a (nints, dimy, dimx) 3D cube
        if outlier_maps is None:
            print(' No outlier maps passed, ignoring outliers.')
            outliers = np.zeros((nint, dimy, dimx))
        else:
            print(' Using outlier map {}'.format(outlier_maps[n]))
            outliers = fits.getdata(outlier_maps[n])
            # If the outlier map is 2D (dimy, dimx) extend to int dimension.
            if np.ndim(outliers) == 2:
                outliers = np.repeat(outliers, nint)
                outliers = outliers.reshape((dimy, dimx, nint))
                outliers = outliers.transpose(2, 0, 1)
        # The outlier map is 0 where good and >0 otherwise. As this will be
        # applied multiplicatively, replace 0s with 1s and others with NaNs.
        outliers = np.where(outliers == 0, 1, np.nan)

        # Read in the main trace mask - a (dimy, dimx) or (3, dimy, dimx)
        # data frame.
        if trace_mask is None:
            print(' No trace mask passed, ignoring the trace.')
            tracemask = np.zeros((3, dimy, dimx))
        else:
            print(' Using trace mask {}.'.format(trace_mask))
            if isinstance(trace_mask, str):
                tracemask = fits.getdata(trace_mask)
            else:
                msg = 'Unrecognized trace_mask file type: {}.' \
                      'Ignoring the trace mask.'.format(type(trace_mask))
                warnings.warn(msg)
                tracemask = np.zeros((3, dimy, dimx))
        # Trace mask may be order specific, or all order combined. Collapse
        # into a combined mask.
        if np.ndim(tracemask) == 3:
            tracemask = tracemask[0].astype(bool) | tracemask[1].astype(bool)\
                        | tracemask[2].astype(bool)
        else:
            tracemask = tracemask
        # Convert into a multiplicative mask of 1s and NaNs.
        tracemask = np.where(tracemask == 0, 1, np.nan)
        # Reshape into (nints, dimy, dimx) format.
        tracemask = np.repeat(tracemask, nint).reshape((dimy, dimx, nint))
        tracemask = tracemask.transpose(2, 0, 1)
        # Combine the two masks.
        outliers = (outliers + tracemask) // 2

        # Initialize output storage arrays.
        dcmap = np.zeros_like(datamodel.data)
        sub, sub_m = np.zeros_like(dcmap), np.zeros_like(dcmap)
        subcorr = np.zeros_like(dcmap)
        # Loop over all integrations to determine the 1/f noise level via a
        # difference image, and correct it.
        for i in tqdm(range(nint)):
            # i counts ints in this particular segment, whereas ii counnts
            # ints from the start of the exposure.
            ii = current_int + i
            # Create two difference images; one to be masked and one not.
            sub[i] = datamodel.data[i] - deepstack * smoothed_wlc[ii]
            sub_m[i] = datamodel.data[i] - deepstack * smoothed_wlc[ii]
            # Since the variable upon which 1/f noise depends is time, treat
            # each group individually.
            for g in range(ngroup):
                # Consider the DQ mask for the group.
                current_outlier = (outliers[i, :, :] + dq[i, g, :, :]) // 2
                # Apply the outlier mask.
                sub_m[i, g, :, :] *= current_outlier
                # FULL frame uses multiple amplifiers and probably has to be
                # treated differently.
                if datamodel.meta.subarray.name == 'FULL':
                    raise NotImplementedError
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    dc = np.nanmedian(sub_m[i, g], axis=0)
                # dc is 1D (dimx) - expand to 2D (dimy, dimx)
                dc2d = np.repeat(dc, dimy).reshape((dimx, dimy))
                dc2d = dc2d.transpose(1, 0)
                # Save the noise map
                dcmap[i, g, :, :] = dc2d
                # Subtract the noise map to create a corrected difference
                # image - mostly for visualization purposes.
                subcorr[i, g, :, :] = sub[i, g, :, :] - dcmap[i, g, :, :]
        current_int += nint

        # Make sure no NaNs are in the DC map
        dcmap = np.where(np.isfinite(dcmap), dcmap, 0)
        # Subtract the DC map from a copy of the data model
        rampmodel_corr = datamodel.copy()
        corr_data = datamodel.data - dcmap
        rampmodel_corr.data = corr_data

        # Save the results if requested.
        if save_results is True:
            # Inital difference image.
            hdu = fits.PrimaryHDU(sub)
            suffix = 'oneoverfstep_diffim.fits'
            hdu.writeto(output_dir + fileroots[n] + suffix,
                        overwrite=True)
            # 1/f noise-corrected difference image.
            hdu = fits.PrimaryHDU(subcorr)
            suffix = 'oneoverfstep_diffimcorr.fits'
            hdu.writeto(output_dir + fileroots[n] + suffix,
                        overwrite=True)
            # DC noise map.
            hdu = fits.PrimaryHDU(dcmap)
            suffix = 'oneoverfstep_noisemap.fits'
            hdu.writeto(output_dir + fileroots[n] + suffix,
                        overwrite=True)
            corrected_rampmodels.append(rampmodel_corr)
            # Corrected ramp model.
            suffix = 'oneoverfstep.fits'
            rampmodel_corr.write(output_dir + fileroots[n] + suffix)

        # Close datamodel for current segment.
        datamodel.close()

    return corrected_rampmodels


def run_stage1(results, background_model, baseline_ints=None,
               smoothed_wlc=None, save_results=True, outlier_maps=None,
               trace_mask=None,  force_redo=False, rejection_threshold=5,
               root_dir='./', output_tag='', occultation_type='transit'):
    """Run the supreme-SPOON Stage 1 pipeline: detector level processing,
    using a combination of official STScI DMS and custom steps. Documentation
    for the official DMS steps can be found here:
    https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_detector1.html

    Parameters
    ----------
    results : DataBowl, list[str]
        List of paths to input uncalibrated datafiles for all segments in an
        exposure, or supreme-SPOON DataBowl object.
    background_model : np.array
        SOSS background model.
    baseline_ints : list[int]
        Integration numbers for transit ingress and egress.
    smoothed_wlc : np.array, None
        Estimate of the out-of-transit normalized light curve.
    save_results : bool
        If True, save results of each step to file.
    outlier_maps : list[str], None
        For improved 1/f noise corecton. List of paths to outlier maps for each
        data segment. Can be 3D (nints, dimy, dimx), or 2D (dimy, dimx) files.
    trace_mask : str, np.array, None
        For improved 1/f noise correcton. Trace mask, or path to file
        containing a trace mask. Should be 3D (norder, dimy, dimx), or 2D
        (dimy, dimx).
    force_redo : bool
        If True, redo steps even if outputs files are already present.
    rejection_threshold : int
        For jump detection; sigma threshold for a pixel to be considered an
        outlier.
    root_dir : str
        Directory from which all relative paths are defined.
    output_tag : str
        Name tag to append to pipeline outputs directory.
    occultation_type : str
        Type of occultation: transit or eclipse.

    Returns
    -------
    results : DataBowl
        Datafiles for each segment processed through Stage 1.
    """

    # ============== DMS Stage 1 ==============
    # Detector level processing.
    print('\n\n**Starting supreme-SPOON Stage 1**')
    print('Detector level processing\n\n')

    if output_tag != '':
        output_tag = '_' + output_tag
    # Create output directories and define output paths.
    utils.verify_path(root_dir + 'pipeline_outputs_directory' + output_tag)
    utils.verify_path(root_dir + 'pipeline_outputs_directory' + output_tag + '/Stage1')
    outdir = root_dir + 'pipeline_outputs_directory' + output_tag + '/Stage1/'

    if not isinstance(results, DataBowl):
        results = DataBowl(results, occultation_type=occultation_type,
                           trace_mask=trace_mask, baseline_ints=baseline_ints,
                           outlier_maps=outlier_maps)

    # ===== Group Scale Step =====
    # Default DMS step.
    step = GroupScaleStep(results, output_dir=outdir)
    step_results = step.run(save_results=save_results, force_redo=force_redo)
    results.datamodels = step_results

    # ===== Data Quality Initialization Step =====
    # Default DMS step.
    step = DQInitStep(results, output_dir=outdir)
    step_results = step.run(save_results=save_results, force_redo=force_redo)
    results.datamodels = step_results

    # ===== Saturation Detection Step =====
    # Default DMS step.
    step = SaturationStep(results, output_dir=outdir)
    step_results = step.run(save_results=save_results, force_redo=force_redo)
    results.datamodels = step_results

    # ===== Superbias Subtraction Step =====
    # Default DMS step.
    step = SuperBiasStep(results, output_dir=outdir)
    step_results = step.run(save_results=save_results, force_redo=force_redo)
    results.datamodels = step_results

    # ===== Reference Pixel Correction Step =====
    # Default DMS step.
    step = RefPixStep(results, output_dir=outdir)
    step_results = step.run(save_results=save_results, force_redo=force_redo)
    results.datamodels = step_results

    # ===== Background Subtraction Step =====
    # Custom DMS step.
    step = BackgroundStep(results, background_model=background_model,
                          output_dir=outdir)
    step_results = step.run(save_results=save_results, force_redo=force_redo)
    results.datamodels = step_results[0]
    results.background_models = step_results[1]

    # ===== 1/f Noise Correction Step =====
    # Custom DMS step.
    step = OneOverFStep(results, baseline_ints=baseline_ints,
                        output_dir=outdir, outlier_maps=outlier_maps,
                        trace_mask=trace_mask, smoothed_wlc=smoothed_wlc)
    step_results = step.run(save_results=save_results, force_redo=force_redo)
    results.datamodels = step_results

    # ===== Linearity Correction Step =====
    # Default DMS step.
    step = LinearityStep(results, output_dir=outdir)
    step_results = step.run(save_results=save_results, force_redo=force_redo)
    results.datamodels = step_results

    # ===== Jump Detection Step =====
    # Default DMS step.
    step = JumpStep(results, output_dir=outdir)
    step_results = step.run(save_results=save_results, force_redo=force_redo,
                            rejection_threshold=rejection_threshold)
    results.datamodels = step_results

    # ===== Ramp Fit Step =====
    # Default DMS step.
    step = RampFitStep(results, output_dir=outdir)
    step_results = step.run(save_results=save_results, force_redo=force_redo)
    results.datamodels = step_results

    # ===== Gain Scale Correcton Step =====
    # Default DMS step.
    step = GainScaleStep(results, output_dir=outdir)
    step_results = step.run(save_results=save_results, force_redo=force_redo)
    results.datamodels = step_results

    return results
