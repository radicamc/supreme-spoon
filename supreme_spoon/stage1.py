#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Jul 21 17:30 2022

@author: MCR

Custom JWST DMS pipeline steps for Stage 1 (detector level processing).
"""

from astropy.io import fits
import bottleneck as bn
import glob
import numpy as np
from scipy.ndimage import median_filter
from tqdm import tqdm
import warnings

from jwst import datamodels
from jwst.pipeline import calwebb_detector1

from supreme_spoon.stage2 import BackgroundStep
from supreme_spoon import utils


class GroupScaleStep:
    """Wrapper around default calwebb_detector1 Group Scale Correction step.
    """

    def __init__(self, input_data, output_dir='./'):
        """Step initializer.
        """

        self.tag = 'groupscalestep.fits'
        self.output_dir = output_dir
        self.datafiles = np.atleast_1d(input_data)
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
                print('Output file {} already exists.'.format(expected_file))
                print('Skipping Group Scale Step.\n')
                res = datamodels.open(expected_file)
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
        """Step initializer.
        """

        self.tag = 'dqinitstep.fits'
        self.output_dir = output_dir
        self.datafiles = np.atleast_1d(input_data)
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
                print('Output file {} already exists.'.format(expected_file))
                print('Skipping Data Quality Initialization Step.\n')
                res = datamodels.open(expected_file)
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
        """Step initializer.
        """

        self.tag = 'saturationstep.fits'
        self.output_dir = output_dir
        self.datafiles = np.atleast_1d(input_data)
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
                print('Output file {} already exists.'.format(expected_file))
                print('Skipping Saturation Detection Step.\n')
                res = datamodels.open(expected_file)
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
        """Step initializer.
        """

        self.tag = 'superbiasstep.fits'
        self.output_dir = output_dir
        self.datafiles = np.atleast_1d(input_data)
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
                print('Output file {} already exists.'.format(expected_file))
                print('Skipping Superbias Subtraction Step.\n')
                res = datamodels.open(expected_file)
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
        """Step initializer.
        """

        self.tag = 'refpixstep.fits'
        self.output_dir = output_dir
        self.datafiles = np.atleast_1d(input_data)
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
                print('Output file {} already exists.'.format(expected_file))
                print('Skipping Reference Pixel Correction Step.\n')
                res = datamodels.open(expected_file)
            # If no output files are detected, run the step.
            else:
                step = calwebb_detector1.refpix_step.RefPixStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results, **kwargs)
            results.append(res)

        return results


class OneOverFStep:
    """Wrapper around custom 1/f Correction Step.
    """

    def __init__(self, input_data, baseline_ints, output_dir='./',
                 smoothed_wlc=None, outlier_maps=None, trace_mask=None,
                 background=None, occultation_type='transit'):
        """Step initializer.
        """

        self.tag = 'oneoverfstep.fits'
        self.output_dir = output_dir
        self.baseline_ints = baseline_ints
        self.smoothed_wlc = smoothed_wlc
        self.trace_mask = trace_mask
        self.outlier_maps = outlier_maps
        self.background = background
        self.occultation_type = occultation_type
        self.datafiles = np.atleast_1d(input_data)
        self.fileroots = utils.get_filename_root(self.datafiles)

    def run(self, even_odd_rows=True, save_results=True, force_redo=False):
        """Method to run the step.
        """

        all_files = glob.glob(self.output_dir + '*')
        do_step = 1
        results = []
        for i in range(len(self.datafiles)):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file not in all_files:
                do_step = 0
                break
            else:
                results.append(datamodels.open(expected_file))
        if do_step == 1 and force_redo is False:
            print('Output files already exist.')
            print('Skipping 1/f Correction Step.\n')
        # If no output files are detected, run the step.
        else:
            results = oneoverfstep(self.datafiles,
                                   baseline_ints=self.baseline_ints,
                                   even_odd_rows=even_odd_rows,
                                   background=self.background,
                                   smoothed_wlc=self.smoothed_wlc,
                                   output_dir=self.output_dir,
                                   save_results=save_results,
                                   outlier_maps=self.outlier_maps,
                                   trace_mask=self.trace_mask,
                                   fileroots=self.fileroots,
                                   occultation_type=self.occultation_type)

        return results


class LinearityStep:
    """Wrapper around default calwebb_detector1 Linearity Correction step.
    """

    def __init__(self, input_data, output_dir='./'):
        """Step initializer.
        """

        self.tag = 'linearitystep.fits'
        self.output_dir = output_dir
        self.datafiles = np.atleast_1d(input_data)
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
                print('Output file {} already exists.'.format(expected_file))
                print('Skipping Linearity Correction Step.\n')
                res = datamodels.open(expected_file)
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
        """Step initializer.
        """

        self.tag = 'jump.fits'
        self.output_dir = output_dir
        self.datafiles = np.atleast_1d(input_data)
        self.fileroots = utils.get_filename_root(self.datafiles)

    def run(self, save_results=True, force_redo=False, rejection_threshold=5,
            ngroup_flag=False, **kwargs):
        """Method to run the step.
        """

        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                print('Output file {} already exists.'.format(expected_file))
                print('Skipping Jump Detection Step.\n')
                results.append(datamodels.open(expected_file))
            # If no output files are detected, proceed.
            else:
                # Get number of groups in the observation - ngroup=2 must be
                # treated in a special way as the default pipeline JumpStep
                # will fail.
                testfile = datamodels.open(self.datafiles[0])
                ngroups = testfile.meta.exposure.ngroups
                testfile.close()
                # for ngroup > 2, use the default JumpStep.
                if ngroups > 2:
                    step = calwebb_detector1.jump_step.JumpStep()
                    res = step.call(segment, output_dir=self.output_dir,
                                    save_results=save_results,
                                    rejection_threshold=rejection_threshold,
                                    maximum_cores='quarter', **kwargs)
                    results.append(res)
                # If ngroup = 2, use a custom temporal domain jump flagging
                # algorithm, which is applied after ramp fitting.
                elif ngroups == 2 and ngroup_flag is False:
                    # If before the RampFitStep, just pass.
                    print('Observation has ngroups=2.')
                    print('Jump detection will be treated after ramp fitting.')
                    ngroup_flag = True
                    results = self.datafiles
                    break
                else:
                    # If after the RampFitStep, run the two group jump
                    # detection.
                    results = two_group_jumpstep(self.datafiles,
                                                 thresh=rejection_threshold,
                                                 fileroots=self.fileroots,
                                                 save_results=save_results,
                                                 output_dir=self.output_dir)
                    break

        return results, ngroup_flag


class RampFitStep:
    """Wrapper around default calwebb_detector1 Ramp Fit step.
    """

    def __init__(self, input_data, output_dir='./'):
        """Step initializer.
        """

        self.tag = 'rampfitstep.fits'
        self.output_dir = output_dir
        self.datafiles = np.atleast_1d(input_data)
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
                print('Output file {} already exists.'.format(expected_file))
                print('Skipping Ramp Fit Step.\n')
                res = datamodels.open(expected_file)
            # If no output files are detected, run the step.
            else:
                step = calwebb_detector1.ramp_fit_step.RampFitStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results,
                                maximum_cores='quarter', **kwargs)[1]
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
        """Step initializer.
        """

        self.tag = 'gainscalestep.fits'
        self.output_dir = output_dir
        self.datafiles = np.atleast_1d(input_data)
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
                print('Output file {} already exists.'.format(expected_file))
                print('Skipping Gain Scale Correction Step.\n')
                res = datamodels.open(expected_file)
            # If no output files are detected, run the step.
            else:
                step = calwebb_detector1.gain_scale_step.GainScaleStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results, **kwargs)
                # Hack to remove jump tag from file name.
                try:
                    res = utils.fix_filenames(res, '_jump_',
                                              self.output_dir)[0]
                    res = datamodels.open(res)
                except IndexError:
                    pass
            results.append(res)

        return results


def oneoverfstep(datafiles, baseline_ints, even_odd_rows=True,
                 background=None, smoothed_wlc=None, output_dir='./',
                 save_results=True, outlier_maps=None, trace_mask=None,
                 fileroots=None, occultation_type='transit'):
    """Custom 1/f correction routine to be applied at the group level. A
    median stack is constructed using all out-of-transit integrations and
    subtracted from each individual integration. The column-wise median of
    this difference image is then subtracted from the original frame to
    correct 1/f noise. Outlier pixels, as well as the trace itself can be
    masked to improve the noise level estimation.

    Parameters
    ----------
    datafiles : array-like[str], array-like[RampModel], array-like[CubeModel]
        List of paths to data files, or RampModels themselves for each segment
        of the TSO. Should be 4D ramps, but 3D rate files can also be accepted.
    baseline_ints : array-like[int]
        Integration numbers of ingress and egress.
    even_odd_rows : bool
        If True, calculate 1/f noise seperately for even and odd numbered rows.
    background : str, array-like[float], None
        Model of background flux.
    smoothed_wlc : array-like[float], None
        Estimate of the normalized light curve.
    output_dir : str
        Directory to which to save results.
    save_results : bool
        If True, save results to disk.
    outlier_maps : array-like[str], None
        List of paths to outlier maps for each data segment. Can be
        3D (nints, dimy, dimx), or 2D (dimy, dimx) files.
    trace_mask : str, None
        Path to file containing a trace mask. Should be 3D (norder, dimy,
        dimx), or 2D (dimy, dimx).
    fileroots : array-like[str], None
        Root names for output files.
    occultation_type : str
        Type of occultation, either 'transit' or 'eclipse'.

    Returns
    -------
    corrected_rampmodels : array-like
        RampModels for each segment, corrected for 1/f noise.
    """

    print('Starting 1/f correction step.')

    # Output directory formatting.
    if output_dir is not None:
        if output_dir[-1] != '/':
            output_dir += '/'

    # Format the baseline frames - either out-of-transit or in-eclipse.
    baseline_ints = utils.format_out_frames(baseline_ints,
                                            occultation_type)

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
    # baseline integrations.
    msg = 'Generating a deep stack for each frame using baseline' \
          ' integrations...'
    print(msg)
    deepstack = utils.make_deepstack(cube[baseline_ints])

    # In order to subtract off the trace as completely as possible, the median
    # stack must be scaled, via the transit curve, to the flux level of each
    # integration.
    # If no lightcurve is provided, estimate it from the current data.
    if smoothed_wlc is None:
        postage = cube[:, -1, 20:60, 1500:1550]
        timeseries = np.nansum(postage, axis=(1, 2))
        timeseries = timeseries / np.nanmedian(timeseries[baseline_ints])
        # Smooth the time series on a timescale of roughly 2%.
        smoothed_wlc = median_filter(timeseries,
                                     int(0.02*np.shape(cube)[0]))

    # Background must be subtracted to accurately subtract off the target
    # trace and isolate 1/f noise. However, the background flux must also be
    # corrected for non-linearity. Therefore, it should be added back after
    # the 1/f is subtracted to be re-subtracted later.
    if background is not None:
        if isinstance(background, str):
            background = np.load(background)

    # Individually treat each segment.
    corrected_rampmodels = []
    current_int = 0
    for n, datamodel in enumerate(data):
        print('Starting segment {} of {}.'.format(n + 1, len(data)))

        # Define the readout setup - can be 4D (recommended) or 3D.
        if np.ndim(datamodel.data) == 4:
            nint, ngroup, dimy, dimx = np.shape(datamodel.data)
        else:
            nint, dimy, dimx = np.shape(datamodel.data)

        # Read in the outlier map - a (nints, dimy, dimx) 3D cube
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
        corr_data = np.copy(datamodel.data)
        # Loop over all integrations to determine the 1/f noise level via a
        # difference image, and correct it.
        for i in tqdm(range(nint)):
            # i counts ints in this particular segment, whereas ii counts
            # ints from the start of the exposure.
            ii = current_int + i
            # Create the difference image.
            sub = datamodel.data[i] - deepstack * smoothed_wlc[ii]
            # Since the variable upon which 1/f noise depends is time, treat
            # each group individually.
            # Apply the outlier mask.
            sub *= outliers[i, :, :]
            # FULL frame uses multiple amplifiers and probably has to be
            # treated differently.
            if datamodel.meta.subarray.name == 'FULL':
                raise NotImplementedError
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                if even_odd_rows is True:
                    # Calculate 1/f scaling seperately for even and odd
                    # rows. This should be taken care of by the RefPixStep,
                    # but it doesn't hurt to do it again.
                    dc = np.zeros_like(sub)
                    # For group-level corrections.
                    if np.ndim(datamodel.data == 4):
                        dc[:, ::2] = bn.nanmedian(sub[:, ::2], axis=1)[:, None, :]
                        dc[:, 1::2] = bn.nanmedian(sub[:, 1::2], axis=1)[:, None, :]
                    # For integration-level corrections.
                    else:
                        dc[::2] = bn.nanmedian(sub[::2], axis=0)[None, :]
                        dc[1::2] = bn.nanmedian(sub[1::2], axis=0)[None, :]
                else:
                    # Single 1/f scaling for all rows.
                    dc = np.zeros_like(sub)
                    # For group-level corrections.
                    if np.ndim(datamodel.data == 4):
                        dc[:, :, :] = bn.nanmedian(sub, axis=1)[:, None, :]
                    # For integration-level corrections.
                    else:
                        dc[:, :] = bn.nanmedian(sub, axis=0)[None, :]
            # Make sure no NaNs are in the DC map
            dc = np.where(np.isfinite(dc), dc, 0)
            corr_data[i] -= dc
        current_int += nint

        # Add back the zodi background.
        if background is not None:
            corr_data += background

        # Store results.
        rampmodel_corr = datamodel.copy()
        rampmodel_corr.data = corr_data
        corrected_rampmodels.append(rampmodel_corr)

        # Save the results if requested.
        if save_results is True:
            suffix = 'oneoverfstep.fits'
            rampmodel_corr.write(output_dir + fileroots[n] + suffix)

        # Close datamodel for current segment.
        datamodel.close()

    return corrected_rampmodels


def two_group_jumpstep(datafiles, window=5, thresh=10, fileroots=None,
                       save_results=True, output_dir='./'):
    """ Jump detection step for ngroup=2 observations. The standard JWST
    pipeline JumpStep fails for these observations as deviations in a linear
    ramp cannot be identified with only 2 groups. This algorithm is based off
    of Nikolov+ (2014) and identifies cosmic ray hits in the temporal domain.
    All jumps are replaced with the median of surrounding integrations.

    Parameters
    ----------
    datafiles : array-like[str], array-like[CubeModel]
        List of paths to data files, or RampModels themselves for each segment
        of the TSO. Should be 3D rate files.
    window : int
        Number of integrations before and after to use for cosmic ray flagging.
    thresh : int
        Sigma threshold for a pixel to be flagged.
    output_dir : str
        Directory to which to save results.
    save_results : bool
        If True, save results to disk.
    fileroots : array-like[str], None
        Root names for output files.

    Returns
    -------
    corrected_rampmodels : array-like[CubeModel]
        Data files corrected for cosmic ray hits.
    """

    print('Starting two-group jump detection step.')

    datafiles = np.atleast_1d(datafiles)
    opened_datafiles = []
    # Load in each of the datafiles.
    for i, file in enumerate(datafiles):
        currentfile = utils.open_filetype(file)
        opened_datafiles.append(currentfile)
        # Make cube of data and DQ flags.
        if i == 0:
            cube = currentfile.data
            dqcube = currentfile.dq
        else:
            cube = np.concatenate([cube, currentfile.data], axis=0)
            dqcube = np.concatenate([dqcube, currentfile.dq], axis=0)
    corrected_rampmodels = opened_datafiles

    nints, dimy, dimx = np.shape(cube)
    # Jump detection algorithm based on Nikolov+ (2014). For each integration,
    # create a difference image using the median of surrounding integrations.
    # Flag all pixels with deviations more than X-sigma as comsic rays hits.
    count = 0
    for i in tqdm(range(nints)):
        # Create a stack of the integrations before and after the current
        # integration.
        up = np.min([nints, i + window + 1])
        low = np.max([0, i - window])
        stack = np.concatenate([cube[low:i], cube[(i + 1):up]])
        # Get median and standard deviation of the stack.
        local_med = np.nanmedian(stack, axis=0)
        local_std = np.nanstd(stack, axis=0)
        # Find deviant pixels.
        ii = np.where(np.abs(cube[i] - local_med) >= thresh * local_std)
        # Replace flagged pixels with the stack median and remove the dq flag.
        cube[i][ii] = local_med[ii]
        dqcube[i][ii] = 0
        count += len(ii[0])
    print(' {} jumps flagged'.format(count))

    current_int = 0
    # Save interpolated data.
    for n, file in enumerate(corrected_rampmodels):
        currentdata = file.data
        nints = np.shape(currentdata)[0]
        file.data = cube[current_int:(current_int + nints)]
        file.dq = dqcube[current_int:(current_int + nints)]
        current_int += nints
        if save_results is True:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                file.write(output_dir + fileroots[n] + 'jump.fits')
        file.close()

    print('Done')

    return corrected_rampmodels


def run_stage1(results, background_model, baseline_ints=None,
               smoothed_wlc=None, save_results=True, outlier_maps=None,
               trace_mask=None,  even_odd_rows=True, force_redo=False,
               rejection_threshold=5, root_dir='./', output_tag='',
               occultation_type='transit', **kwargs):
    """Run the supreme-SPOON Stage 1 pipeline: detector level processing,
    using a combination of official STScI DMS and custom steps. Documentation
    for the official DMS steps can be found here:
    https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_detector1.html

    Parameters
    ----------
    results : array-like[str]
        List of paths to input uncalibrated datafiles for all segments in an
        exposure.
    background_model : array-like[float]
        SOSS background model.
    baseline_ints : array-like[int]
        Integration numbers for transit ingress and egress.
    smoothed_wlc : array-like[float], None
        Estimate of the out-of-transit normalized light curve.
    save_results : bool
        If True, save results of each step to file.
    outlier_maps : array-like[str], None
        For improved 1/f noise corecton. List of paths to outlier maps for each
        data segment. Can be 3D (nints, dimy, dimx), or 2D (dimy, dimx) files.
    trace_mask : str, array-like[bool], None
        For improved 1/f noise correcton. Trace mask, or path to file
        containing a trace mask. Should be 3D (norder, dimy, dimx), or 2D
        (dimy, dimx).
    even_odd_rows : bool
        If True, calculate 1/f noise seperately for even and odd numbered rows.
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
    results : array-like[RampModel]
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

    # ===== Group Scale Step =====
    # Default DMS step.
    if 'GroupScaleStep' in kwargs.keys():
        step_kwargs = kwargs['GroupScaleStep']
    else:
        step_kwargs = {}
    step = GroupScaleStep(results, output_dir=outdir)
    results = step.run(save_results=save_results, force_redo=force_redo,
                       **step_kwargs)

    # ===== Data Quality Initialization Step =====
    # Default DMS step.
    if 'DQInitStep' in kwargs.keys():
        step_kwargs = kwargs['DQInitStep']
    else:
        step_kwargs = {}
    step = DQInitStep(results, output_dir=outdir)
    results = step.run(save_results=save_results, force_redo=force_redo,
                       **step_kwargs)

    # ===== Saturation Detection Step =====
    # Default DMS step.
    if 'SaturationStep' in kwargs.keys():
        step_kwargs = kwargs['SaturationStep']
    else:
        step_kwargs = {}
    step = SaturationStep(results, output_dir=outdir)
    results = step.run(save_results=save_results, force_redo=force_redo,
                       **step_kwargs)

    # ===== Superbias Subtraction Step =====
    # Default DMS step.
    if 'SuperBiasStep' in kwargs.keys():
        step_kwargs = kwargs['SuperBiasStep']
    else:
        step_kwargs = {}
    step = SuperBiasStep(results, output_dir=outdir)
    results = step.run(save_results=save_results, force_redo=force_redo,
                       **step_kwargs)

    # ===== Reference Pixel Correction Step =====
    # Default DMS step.
    if 'RefPixStep' in kwargs.keys():
        step_kwargs = kwargs['RefPixStep']
    else:
        step_kwargs = {}
    step = RefPixStep(results, output_dir=outdir)
    results = step.run(save_results=save_results, force_redo=force_redo,
                       **step_kwargs)

    # ===== Background Subtraction Step =====
    # Custom DMS step - imported from Stage2.
    step = BackgroundStep(results, background_model=background_model,
                          output_dir=outdir)
    results = step.run(save_results=save_results, force_redo=force_redo)
    results, background_model = results

    # ===== 1/f Noise Correction Step =====
    # Custom DMS step.
    step = OneOverFStep(results, baseline_ints=baseline_ints,
                        output_dir=outdir, outlier_maps=outlier_maps,
                        trace_mask=trace_mask, smoothed_wlc=smoothed_wlc,
                        background=background_model,
                        occultation_type=occultation_type)
    results = step.run(even_odd_rows=even_odd_rows, save_results=save_results,
                       force_redo=force_redo)

    # ===== Linearity Correction Step =====
    # Default DMS step.
    if 'LinearityStep' in kwargs.keys():
        step_kwargs = kwargs['LinearityStep']
    else:
        step_kwargs = {}
    step = LinearityStep(results, output_dir=outdir)
    results = step.run(save_results=save_results, force_redo=force_redo,
                       **step_kwargs)

    # ===== Jump Detection Step =====
    # Default DMS step.
    if 'JumpStep' in kwargs.keys():
        step_kwargs = kwargs['JumpStep']
    else:
        step_kwargs = {}
    step = JumpStep(results, output_dir=outdir)
    step_results = step.run(save_results=save_results, force_redo=force_redo,
                            rejection_threshold=rejection_threshold,
                            **step_kwargs)
    results, ngroup_flag = step_results

    # ===== Ramp Fit Step =====
    # Default DMS step.
    if 'RampFitStep' in kwargs.keys():
        step_kwargs = kwargs['RampFitStep']
    else:
        step_kwargs = {}
    step = RampFitStep(results, output_dir=outdir)
    results = step.run(save_results=save_results, force_redo=force_redo,
                       **step_kwargs)

    # ===== Jump Detection Step =====
    # Custom DMS step - specifically for ngroup=2.
    if ngroup_flag is True:
        step = JumpStep(results, output_dir=outdir)
        results = step.run(save_results=save_results, force_redo=force_redo,
                           rejection_threshold=rejection_threshold,
                           ngroup_flag=True)[0]

    # ===== Gain Scale Correcton Step =====
    # Default DMS step.
    if 'GainScaleStep' in kwargs.keys():
        step_kwargs = kwargs['GainScaleStep']
    else:
        step_kwargs = {}
    step = GainScaleStep(results, output_dir=outdir)
    results = step.run(save_results=save_results, force_redo=force_redo,
                       **step_kwargs)

    return results
