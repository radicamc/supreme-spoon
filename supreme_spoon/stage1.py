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
import os
from scipy.ndimage import median_filter
from tqdm import tqdm
import warnings

from jwst import datamodels
from jwst.pipeline import calwebb_detector1

from supreme_spoon.stage2 import BackgroundStep
from supreme_spoon import utils, plotting
from supreme_spoon.utils import fancyprint


class GroupScaleStep:
    """Wrapper around default calwebb_detector1 Group Scale Correction step.
    """

    def __init__(self, input_data, output_dir):
        """Step initializer.
        """

        self.tag = 'groupscalestep.fits'
        self.output_dir = output_dir
        self.datafiles = utils.sort_datamodels(input_data)
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
                fancyprint('Skipping Group Scale Step.\n')
                res = expected_file
            # If no output files are detected, run the step.
            else:
                step = calwebb_detector1.group_scale_step.GroupScaleStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results, **kwargs)
                # Verify that filename is correct.
                if save_results is True:
                    current_name = self.output_dir + res.meta.filename
                    if expected_file != current_name:
                        res.close()
                        os.rename(current_name, expected_file)
                        res = datamodels.open(expected_file)
            results.append(res)

        return results


class DQInitStep:
    """Wrapper around default calwebb_detector1 Data Quality Initialization
    step.
    """

    def __init__(self, input_data, output_dir):
        """Step initializer.
        """

        self.tag = 'dqinitstep.fits'
        self.output_dir = output_dir
        self.datafiles = utils.sort_datamodels(input_data)
        self.fileroots = utils.get_filename_root(self.datafiles)

    def run(self, save_results=True, force_redo=False, deepframe=None,
            **kwargs):
        """Method to run the step.
        """

        results = []
        all_files = glob.glob(self.output_dir + '*')
        hot_pix = None
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                fancyprint('File {} already exists.'.format(expected_file))
                fancyprint('Skipping Data Quality Initialization Step.\n')
                res = expected_file
            # If no output files are detected, run the step.
            else:
                step = calwebb_detector1.dq_init_step.DQInitStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results, **kwargs)
                # If a deep frame is passed, use it to search for and flag
                # additional hot pixels not in the default map.
                if deepframe is not None:
                    res, hot_pix = utils.flag_hot_pixels(res,
                                                         deepframe=deepframe,
                                                         hot_pix=hot_pix)
                    # Overwite the previous edition.
                    res.save(expected_file)
                # Verify that filename is correct.
                if save_results is True:
                    current_name = self.output_dir + res.meta.filename
                    if expected_file != current_name:
                        res.close()
                        os.rename(current_name, expected_file)
                        res = datamodels.open(expected_file)
            results.append(res)

        return results


class SaturationStep:
    """Wrapper around default calwebb_detector1 Saturation Detection step.
    """

    def __init__(self, input_data, output_dir):
        """Step initializer.
        """

        self.tag = 'saturationstep.fits'
        self.output_dir = output_dir
        self.datafiles = utils.sort_datamodels(input_data)
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
                fancyprint('Skipping Saturation Detection Step.\n')
                res = expected_file
            # If no output files are detected, run the step.
            else:
                step = calwebb_detector1.saturation_step.SaturationStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results, **kwargs)
                # Verify that filename is correct.
                if save_results is True:
                    current_name = self.output_dir + res.meta.filename
                    if expected_file != current_name:
                        res.close()
                        os.rename(current_name, expected_file)
                        res = datamodels.open(expected_file)
            results.append(res)

        return results


class SuperBiasStep:
    """Wrapper around default calwebb_detector1 Super Bias Subtraction step.
    """

    def __init__(self, input_data, output_dir):
        """Step initializer.
        """

        self.tag = 'superbiasstep.fits'
        self.output_dir = output_dir
        self.datafiles = utils.sort_datamodels(input_data)
        self.fileroots = utils.get_filename_root(self.datafiles)

    def run(self, save_results=True, force_redo=False, do_plot=False,
            **kwargs):
        """Method to run the step.
        """

        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                fancyprint('File {} already exists.'.format(expected_file))
                fancyprint('Skipping Superbias Subtraction Step.\n')
                res = expected_file
                do_plot = False
            # If no output files are detected, run the step.
            else:
                step = calwebb_detector1.superbias_step.SuperBiasStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results, **kwargs)
                # Verify that filename is correct.
                if save_results is True:
                    current_name = self.output_dir + res.meta.filename
                    if expected_file != current_name:
                        res.close()
                        os.rename(current_name, expected_file)
                        res = datamodels.open(expected_file)
            results.append(res)
        # Do step plot if requested.
        if do_plot is True:
            plot_file = self.output_dir + self.tag.replace('fits', 'pdf')
            plotting.make_superbias_plot(results, outfile=plot_file)

        return results


class RefPixStep:
    """Wrapper around default calwebb_detector1 Reference Pixel Correction
    step.
    """

    def __init__(self, input_data, output_dir):
        """Step initializer.
        """

        self.tag = 'refpixstep.fits'
        self.output_dir = output_dir
        self.datafiles = utils.sort_datamodels(input_data)
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
                fancyprint('Skipping Reference Pixel Correction Step.\n')
                res = expected_file
            # If no output files are detected, run the step.
            else:
                step = calwebb_detector1.refpix_step.RefPixStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results, **kwargs)
                # Verify that filename is correct.
                if save_results is True:
                    current_name = self.output_dir + res.meta.filename
                    if expected_file != current_name:
                        res.close()
                        os.rename(current_name, expected_file)
                        res = datamodels.open(expected_file)
            results.append(res)

        return results


class OneOverFStep:
    """Wrapper around custom 1/f Correction Step.
    """

    def __init__(self, input_data, baseline_ints, output_dir,
                 smoothed_wlc=None, pixel_masks=None, background=None,
                 occultation_type='transit'):
        """Step initializer.
        """

        self.tag = 'oneoverfstep.fits'
        self.output_dir = output_dir
        self.baseline_ints = baseline_ints
        self.smoothed_wlc = smoothed_wlc
        self.pixel_masks = pixel_masks
        self.background = background
        self.occultation_type = occultation_type
        self.datafiles = utils.sort_datamodels(input_data)
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
            fancyprint('Output files already exist.')
            fancyprint('Skipping 1/f Correction Step.\n')
        # If no output files are detected, run the step.
        else:
            results = oneoverfstep(self.datafiles,
                                   baseline_ints=self.baseline_ints,
                                   even_odd_rows=even_odd_rows,
                                   background=self.background,
                                   smoothed_wlc=self.smoothed_wlc,
                                   output_dir=self.output_dir,
                                   save_results=save_results,
                                   pixel_masks=self.pixel_masks,
                                   fileroots=self.fileroots,
                                   occultation_type=self.occultation_type)

        return results


class LinearityStep:
    """Wrapper around default calwebb_detector1 Linearity Correction step.
    """

    def __init__(self, input_data, output_dir):
        """Step initializer.
        """

        self.tag = 'linearitystep.fits'
        self.output_dir = output_dir
        self.datafiles = utils.sort_datamodels(input_data)
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
                fancyprint('Skipping Linearity Correction Step.\n')
                res = expected_file
            # If no output files are detected, run the step.
            else:
                step = calwebb_detector1.linearity_step.LinearityStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results, **kwargs)
                # Verify that filename is correct.
                if save_results is True:
                    current_name = self.output_dir + res.meta.filename
                    if expected_file != current_name:
                        res.close()
                        os.rename(current_name, expected_file)
                        res = datamodels.open(expected_file)
            results.append(res)

        return results


class JumpStep:
    """Wrapper around default calwebb_detector1 Jump Detection step.
    """

    def __init__(self, input_data, output_dir):
        """Step initializer.
        """

        self.tag = 'jump.fits'
        self.output_dir = output_dir
        self.datafiles = utils.sort_datamodels(input_data)
        self.fileroots = utils.get_filename_root(self.datafiles)

    def run(self, save_results=True, force_redo=False, rejection_threshold=15,
            flag_in_time=False, time_rejection_threshold=10, time_window=10,
            **kwargs):
        """Method to run the step.
        """

        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                fancyprint('File {} already exists.'.format(expected_file))
                fancyprint('Skipping Jump Detection Step.\n')
                results.append(datamodels.open(expected_file))
            # If no output files are detected, proceed.
            else:
                # Get number of groups in the observation - ngroup=2 must be
                # treated in a special way as the default pipeline JumpStep
                # will fail.
                testfile = datamodels.open(self.datafiles[0])
                ngroups = testfile.meta.exposure.ngroups
                testfile.close()
                # For ngroup > 2, default JumpStep can be used.
                if ngroups > 2:
                    step = calwebb_detector1.jump_step.JumpStep()
                    res = step.call(segment, output_dir=self.output_dir,
                                    save_results=save_results,
                                    rejection_threshold=rejection_threshold,
                                    maximum_cores='quarter', **kwargs)
                # Time domain jump step must be run for ngroup=2.
                else:
                    res = segment
                    flag_in_time = True
                # Do time-domain flagging.
                if flag_in_time is True:
                    res = jumpstep_in_time(res, window=time_window,
                                           thresh=time_rejection_threshold,
                                           fileroot=self.fileroots[i],
                                           save_results=save_results,
                                           output_dir=self.output_dir)
                # Verify that filename is correct.
                if save_results is True:
                    current_name = self.output_dir + res.meta.filename
                    if expected_file != current_name:
                        res.close()
                        os.rename(current_name, expected_file)
                        res = datamodels.open(expected_file)
                results.append(res)

        return results


class RampFitStep:
    """Wrapper around default calwebb_detector1 Ramp Fit step.
    """

    def __init__(self, input_data, output_dir):
        """Step initializer.
        """

        self.tag = 'rampfitstep.fits'
        self.output_dir = output_dir
        self.datafiles = utils.sort_datamodels(input_data)
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
                fancyprint('Skipping Ramp Fit Step.\n')
                res = expected_file
            # If no output files are detected, run the step.
            else:
                step = calwebb_detector1.ramp_fit_step.RampFitStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results,
                                maximum_cores='quarter', **kwargs)[1]
                if save_results is True:
                    # Store flags for use in 1/f correction.
                    flags = res.dq
                    flags[flags != 0] = 1  # Convert to binary mask.
                    hdu = fits.PrimaryHDU(flags)
                    outfile = self.output_dir + self.fileroots[i] + 'pixelflags.fits'
                    hdu.writeto(outfile, overwrite=True)
                    # Remove rate file because we don't need it and I don't
                    # like having extra files.
                    rate = res.meta.filename.replace('_1_ramp', '_0_ramp')
                    os.remove(self.output_dir + rate)
                    # Verify that filename is correct.
                    current_name = self.output_dir + res.meta.filename
                    if expected_file != current_name:
                        res.close()
                        os.rename(current_name, expected_file)
                        res = datamodels.open(expected_file)
            results.append(res)

        return results


class GainScaleStep:
    """Wrapper around default calwebb_detector1 Gain Scale Correction step.
    """

    def __init__(self, input_data, output_dir):
        """Step initializer.
        """

        self.tag = 'gainscalestep.fits'
        self.output_dir = output_dir
        self.datafiles = utils.sort_datamodels(input_data)
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
                fancyprint('Skipping Gain Scale Correction Step.\n')
                res = expected_file
            # If no output files are detected, run the step.
            else:
                step = calwebb_detector1.gain_scale_step.GainScaleStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results, **kwargs)
                # Verify that filename is correct.
                if save_results is True:
                    current_name = self.output_dir + res.meta.filename
                    if expected_file != current_name:
                        res.close()
                        os.rename(current_name, expected_file)
                        res = datamodels.open(expected_file)
            results.append(res)

        return results


def oneoverfstep(datafiles, baseline_ints, even_odd_rows=True,
                 background=None, smoothed_wlc=None, output_dir=None,
                 save_results=True, pixel_masks=None, fileroots=None,
                 occultation_type='transit'):
    """Custom 1/f correction routine to be applied at the group level. A
    median stack is constructed using all out-of-transit integrations and
    subtracted from each individual integration. The column-wise median of
    this difference image is then subtracted from the original frame to
    correct 1/f noise. Outlier pixels, background contaminants, and the target
    trace itself can (should) be masked to improve the estimation.

    Parameters
    ----------
    datafiles : array-like[str], array-like[RampModel], array-like[CubeModel]
        List of paths to data files, or datamodels themselves for each segment
        of the TSO. Should be 4D ramps, but 3D rate files are also accepted.
    baseline_ints : array-like[int]
        Integration numbers of ingress and egress.
    even_odd_rows : bool
        If True, calculate 1/f noise seperately for even and odd numbered rows.
    background : str, array-like[float], None
        Model of background flux.
    smoothed_wlc : array-like[float], str, None
        Estimate of the normalized light curve, or path to file containing it.
    output_dir : str, None
        Directory to which to save results. Only necessary if saving results.
    save_results : bool
        If True, save results to disk.
    pixel_masks : array-like[str], None
        List of paths to maps of pixels to mask for each data segment. Can be
        3D (nints, dimy, dimx), or 2D (dimy, dimx).
    fileroots : array-like[str], None
        Root names for output files. Only necessary if saving results.
    occultation_type : str
        Type of occultation, either 'transit' or 'eclipse'.

    Returns
    -------
    corrected_rampmodels : array-like[CubeModel]
        RampModels for each segment, corrected for 1/f noise.
    """

    fancyprint('Starting 1/f correction step.')

    # If saving results, ensure output directory and fileroots are provided.
    if save_results is True:
        assert output_dir is not None
        assert fileroots is not None
        # Output directory formatting.
        if output_dir[-1] != '/':
            output_dir += '/'

    # Format the baseline frames - either out-of-transit or in-eclipse.
    baseline_ints = utils.format_out_frames(baseline_ints,
                                            occultation_type)

    datafiles = np.atleast_1d(datafiles)
    # If outlier maps are passed, ensure that there is one for each segment.
    if pixel_masks is not None:
        pixel_masks = np.atleast_1d(pixel_masks)
        if len(pixel_masks) == 1:
            pixel_masks = [pixel_masks[0] for d in datafiles]

    data = []
    # Load in datamodels from all segments.
    for i, file in enumerate(datafiles):
        currentfile = utils.open_filetype(file)
        # FULL frame uses multiple amplifiers and probably has to be
        # treated differently. Break if we encounted a FULL frame exposure.
        if currentfile.meta.subarray.name == 'FULL':
            raise NotImplementedError
        data.append(currentfile)
        # To create the deepstack, join all segments together.
        if i == 0:
            cube = currentfile.data
        else:
            cube = np.concatenate([cube, currentfile.data], axis=0)

    # Generate the 3D deep stack (ngroup, dimy, dimx) using only
    # baseline integrations.
    msg = 'Generating a deep stack for each group using baseline' \
          ' integrations...'
    fancyprint(msg)
    deepstack = utils.make_deepstack(cube[baseline_ints])

    # In order to subtract off the trace as completely as possible, the median
    # stack must be scaled, via the transit curve, to the flux level of each
    # integration.
    # Try to open light curve file. If not, estimate it from data.
    if isinstance(smoothed_wlc, str):
        try:
            smoothed_wlc = np.load(smoothed_wlc)
        except (ValueError, FileNotFoundError):
            msg = 'Light curve file cannot be opened. ' \
                  'It will be estimated from current data.'
            fancyprint(msg, msg_type='WARNING')
            smoothed_wlc = None
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
    # the 1/f is subtracted, in order to be re-subtracted later.
    if background is not None:
        if isinstance(background, str):
            background = np.load(background)

    # Individually treat each segment.
    corrected_rampmodels = []
    current_int = 0
    for n, datamodel in enumerate(data):
        fancyprint('Starting segment {} of {}.'.format(n+1, len(data)))

        # Define the readout setup - can be 4D (recommended) or 3D.
        if np.ndim(datamodel.data) == 4:
            nint, ngroup, dimy, dimx = np.shape(datamodel.data)
        else:
            nint, dimy, dimx = np.shape(datamodel.data)

        # Read in the outlier map - a (nints, dimy, dimx) 3D cube
        if pixel_masks is None:
            fancyprint(' No outlier maps passed, ignoring outliers.')
            outliers = np.zeros((nint, dimy, dimx))
        else:
            fancyprint(' Using outlier map {}'.format(pixel_masks[n]))
            outliers = fits.getdata(pixel_masks[n])
            # If the outlier map is 2D (dimy, dimx) extend to int dimension.
            if np.ndim(outliers) == 2:
                outliers = np.repeat(outliers, nint)
                outliers = outliers.reshape((dimy, dimx, nint))
                outliers = outliers.transpose(2, 0, 1)
        # The outlier map is 0 where good and >0 otherwise. As this will be
        # applied multiplicatively, replace 0s with 1s and others with NaNs.
        outliers = np.where(outliers == 0, 1, np.nan)

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
            # Apply the outlier mask.
            sub *= outliers[i, :, :]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                if even_odd_rows is True:
                    # Calculate 1/f scaling seperately for even and odd rows.
                    # This *should* be taken care of by the RefPixStep, but it
                    # doesn't hurt to do it again.
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
        current_int += nint  # Increment the total integration counter.

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


def jumpstep_in_time(datafile, window=10, thresh=10, fileroot=None,
                     save_results=True, output_dir=None):
    """Jump detection step in the temporal domain. This algorithm is based off
    of Nikolov+ (2014) and identifies cosmic ray hits in the temporal domain.
    All jumps for ngroup<=2 are replaced with the median of surrounding
    integrations, whereas jumps for ngroup>3 are flagged.

    Parameters
    ----------
    datafile : str, RampModel
        Path to data file, or RampModel itself for a segment of the TSO.
        Should be 4D ramp.
    window : int
        Number of integrations before and after to use for cosmic ray flagging.
    thresh : int
        Sigma threshold for a pixel to be flagged.
    output_dir : str
        Directory to which to save results.
    save_results : bool
        If True, save results to disk.
    fileroot : str, None
        Root name for output file.

    Returns
    -------
    datafile : RampModel
        Data file corrected for cosmic ray hits.
    """

    fancyprint('Starting time-domain jump detection step.')

    # If saving results, ensure output directory and fileroots are provided.
    if save_results is True:
        assert output_dir is not None
        assert fileroot is not None
        # Output directory formatting.
        if output_dir[-1] != '/':
            output_dir += '/'

    # Load in the datafile.
    datafile = utils.open_filetype(datafile)
    cube = datafile.data
    dqcube = datafile.groupdq

    nints, ngroups, dimy, dimx = np.shape(cube)
    # Jump detection algorithm based on Nikolov+ (2014). For each integration,
    # create a difference image using the median of surrounding integrations.
    # Flag all pixels with deviations more than X-sigma as comsic rays hits.
    count, interp = 0, 0
    for i in tqdm(range(nints)):
        # Create a stack of the integrations before and after the current
        # integration.
        up = np.min([nints, i + window + 1])
        low = np.max([0, i - window])
        stack = np.concatenate([cube[low:i], cube[(i + 1):up]])
        # Get median and standard deviation of the stack.
        local_med = np.nanmedian(stack, axis=0)
        local_std = np.nanstd(stack, axis=0)
        for g in range(ngroups):
            # Find deviant pixels.
            ii = np.where(np.abs(cube[i, g] - local_med[g]) >= thresh * local_std[g])
            # If ngroup<=2, replace the pixel with the stack median so that a
            # ramp can still be fit.
            if g < 2:
                # Do not want to interpolate pixels which are flagged for
                # another reason, so only select good pixels or those which
                # are flagged for jumps.
                jj = np.where((dqcube[i, g][ii] == 0) | (dqcube[i, g][ii] == 4))
                # Replace these pixels with the stack median and remove the
                # dq flag.
                cube[i, g][ii][jj] = local_med[g][ii][jj]
                dqcube[i, g][ii][jj] = 0
                interp += len(jj[0])
            # If ngroup>2, flag the pixel as having a jump.
            else:
                # Want to ignore pixels which are already flagged for a jump.
                jj = np.where(utils.get_dq_flag_metrics(dqcube[i, g], ['JUMP_DET']) == 1)
                alrdy_flg = np.ones_like(dqcube[i, g]).astype(bool)
                alrdy_flg[jj] = False
                new_flg = np.zeros_like(dqcube[i, g]).astype(bool)
                new_flg[ii] = True
                to_flag = new_flg & alrdy_flg
                # Add the jump detection flag.
                dqcube[i, g][to_flag] += 4
                count += int(np.sum(to_flag))
    fancyprint(' {} jumps flagged'.format(count))
    fancyprint(' and {} interpolated'.format(interp))

    datafile.data = cube
    datafile.groupdq = dqcube
    if save_results is True:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            datafile.write(output_dir + fileroot + 'jump.fits')
    datafile.close()

    fancyprint('Done')

    return datafile


def run_stage1(results, background_model, baseline_ints=None,
               smoothed_wlc=None, save_results=True, outlier_maps=None,
               trace_mask=None,  even_odd_rows=True, force_redo=False,
               rejection_threshold=5, root_dir='./', output_tag='',
               occultation_type='transit', skip_steps=None, **kwargs):
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
    skip_steps : array-like[str], None
        Step names to skip (if any).

    Returns
    -------
    results : array-like[RampModel]
        Datafiles for each segment processed through Stage 1.
    """

    # ============== DMS Stage 1 ==============
    # Detector level processing.
    fancyprint('\n\n**Starting supreme-SPOON Stage 1**')
    fancyprint('Detector level processing\n\n')

    if output_tag != '':
        output_tag = '_' + output_tag
    # Create output directories and define output paths.
    utils.verify_path(root_dir + 'pipeline_outputs_directory' + output_tag)
    outdir = root_dir + 'pipeline_outputs_directory' + output_tag + '/Stage1/'
    utils.verify_path(outdir)

    if skip_steps is None:
        skip_steps = []

    # ===== Group Scale Step =====
    # Default DMS step.
    if 'GroupScaleStep' not in skip_steps:
        if 'GroupScaleStep' in kwargs.keys():
            step_kwargs = kwargs['GroupScaleStep']
        else:
            step_kwargs = {}
        step = GroupScaleStep(results, output_dir=outdir)
        results = step.run(save_results=save_results, force_redo=force_redo,
                           **step_kwargs)

    # ===== Data Quality Initialization Step =====
    # Default DMS step.
    if 'DQInitStep' not in skip_steps:
        if 'DQInitStep' in kwargs.keys():
            step_kwargs = kwargs['DQInitStep']
        else:
            step_kwargs = {}
        step = DQInitStep(results, output_dir=outdir)
        results = step.run(save_results=save_results, force_redo=force_redo,
                           **step_kwargs)

    # ===== Saturation Detection Step =====
    # Default DMS step.
    if 'SaturationStep' not in skip_steps:
        if 'SaturationStep' in kwargs.keys():
            step_kwargs = kwargs['SaturationStep']
        else:
            step_kwargs = {}
        step = SaturationStep(results, output_dir=outdir)
        results = step.run(save_results=save_results, force_redo=force_redo,
                           **step_kwargs)

    # ===== Superbias Subtraction Step =====
    # Default DMS step.
    if 'SuperBiasStep' not in skip_steps:
        if 'SuperBiasStep' in kwargs.keys():
            step_kwargs = kwargs['SuperBiasStep']
        else:
            step_kwargs = {}
        step = SuperBiasStep(results, output_dir=outdir)
        results = step.run(save_results=save_results, force_redo=force_redo,
                           **step_kwargs)

    # ===== Reference Pixel Correction Step =====
    # Default DMS step.
    if 'RefPixStep' not in skip_steps:
        if 'RefPixStep' in kwargs.keys():
            step_kwargs = kwargs['RefPixStep']
        else:
            step_kwargs = {}
        step = RefPixStep(results, output_dir=outdir)
        results = step.run(save_results=save_results, force_redo=force_redo,
                           **step_kwargs)

    if 'OneOverFStep' not in skip_steps:
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
        results = step.run(even_odd_rows=even_odd_rows,
                           save_results=save_results, force_redo=force_redo)

    # ===== Linearity Correction Step =====
    # Default DMS step.
    if 'LinearityStep' not in skip_steps:
        if 'LinearityStep' in kwargs.keys():
            step_kwargs = kwargs['LinearityStep']
        else:
            step_kwargs = {}
        step = LinearityStep(results, output_dir=outdir)
        results = step.run(save_results=save_results, force_redo=force_redo,
                           **step_kwargs)

    # ===== Jump Detection Step =====
    # Default DMS step.
    if 'JumpStep' not in skip_steps:
        if 'JumpStep' in kwargs.keys():
            step_kwargs = kwargs['JumpStep']
        else:
            step_kwargs = {}
        step = JumpStep(results, output_dir=outdir)
        step_results = step.run(save_results=save_results,
                                force_redo=force_redo,
                                rejection_threshold=rejection_threshold,
                                **step_kwargs)
        results, ngroup_flag = step_results

    # ===== Ramp Fit Step =====
    # Default DMS step.
    if 'RampFitStep' not in skip_steps:
        if 'RampFitStep' in kwargs.keys():
            step_kwargs = kwargs['RampFitStep']
        else:
            step_kwargs = {}
        step = RampFitStep(results, output_dir=outdir)
        results = step.run(save_results=save_results, force_redo=force_redo,
                           **step_kwargs)

    # ===== Jump Detection Step =====
    # Custom DMS step - specifically for ngroup=2.
    if 'JumpStep' not in skip_steps:
        if ngroup_flag is True:
            step = JumpStep(results, output_dir=outdir)
            results = step.run(save_results=save_results,
                               force_redo=force_redo,
                               rejection_threshold=rejection_threshold,
                               ngroup_flag=True)[0]

    # ===== Gain Scale Correcton Step =====
    # Default DMS step.
    if 'GainScaleStep' not in skip_steps:
        if 'GainScaleStep' in kwargs.keys():
            step_kwargs = kwargs['GainScaleStep']
        else:
            step_kwargs = {}
        step = GainScaleStep(results, output_dir=outdir)
        results = step.run(save_results=save_results, force_redo=force_redo,
                           **step_kwargs)

    return results
