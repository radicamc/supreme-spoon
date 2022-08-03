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
from tqdm import tqdm
import warnings

from jwst import datamodels
from jwst.pipeline import calwebb_detector1

from supreme_spoon import utils, plotting


def backgroundstep(datafiles, background_model, output_dir=None,
                   save_results=True, show_plots=False):
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
    datafiles : list[str], list[CubeModel]
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

    Returns
    -------
    results : list[CubeModel]
        Input data segments, corrected for the background.
    """

    print('Starting custom background subtraction step.')
    # Output directory formatting.
    if output_dir is not None:
        if output_dir[-1] != '/':
            output_dir += '/'

    datafiles = np.atleast_1d(datafiles)
    opened_datafiles = []
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
    deepstack = bn.nanmedian(cube, axis=0)
    ngroup, dimy, dimx = np.shape(deepstack)

    model_scaled = np.zeros_like(deepstack)
    for i in range(ngroup):
        # Usng a region in the top left corner where influence from the traces is
        # minimal, calculate the scaling of the model background to the median
        # stack.
        bkg_ratio = deepstack[i, 210:250, 500:800] / background_model[210:250, 500:800]
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
    for currentfile in datafiles:
        # Get file name root.
        old_filename = currentfile.meta.filename
        to_remove = old_filename.split('_')[-1]
        fileroot = old_filename.split(to_remove)[0]

        # Subtract the scaled background model.
        data_backsub = currentfile.data - model_scaled
        currentfile.data = data_backsub

        # Save the results to file if requested.
        if save_results is True:
            # Scaled model background.
            hdu = fits.PrimaryHDU(model_scaled)
            hdu.writeto(output_dir + fileroot + 'background.fits',
                        overwrite=True)
            # Background subtracted data.
            currentfile.write(output_dir + fileroot + 'backgroundstep.fits')

        # Show background scaling plot if requested.
        if show_plots is True:
            plotting.do_backgroundsubtraction_plot(currentfile.data[:, -1],
                                                   background_model,
                                                   scale_factor)
        results.append(currentfile)
        currentfile.close()

    return results


def oneoverfstep_ints(datafiles, scaling, out_frames, output_dir=None, save_results=True,
                 outlier_maps=None, trace_mask=None, trace_mask2=None,
                 use_dq=True):
    """Custom 1/f correction routine to be applied at the group level. A
    median stack is constructed using all integrations and subtracted from
    each individual integration. The column-wise median of this difference
    image is then subtracted from the original frame to correct 1/f noise.
    Outlier pixels, as well as the trace itself can be masked to improve the
    noise level estimation. Furthermore, as 1/f noise can also vary with row,
    a column-wise median evaluated in a window around the first and second
    order traces can also be subtracted.

    Parameters
    ----------
    datafiles : list[str], or list[RampModel]
        List of paths to data files, or RampModels themselves for each segment
        of the TSO. Should be 4D ramps and not rate files.
    output_dir : str, None
        Directory to which to save results.
    save_results : bool
        If True, save results to disk.
    outlier_maps : list[str], None
        List of paths to outlier maps for each data segment. Can be
        3D (nints, dimy, dimx), or 2D (dimy, dimx) files.
    trace_mask : str, np.array, None
        Trace mask, or path to file containing a trace mask. Should be 3D
        (norder, dimy, dimx), or 2D (dimy, dimx), although if a "window"
        median subtraction is to be performed, a 3D file is required.
    trace_mask2 : str, np.array, None
        Trace mask, or path to file containing a trace mask. Should be 3D
        (norder, dimy, dimx). If provided, a median evaluated between
        trace_mask and trace_mask2 will also be subtracted.
    use_dq : bool
        If True, mask all pixels currently flagged in the DQ array.

    Returns
    -------
    corrected_rampmodels : list
        RampModels for each segment, corrected for 1/f noise.
    """

    print('Starting custom 1/f correction step.')

    out_frames = np.abs(out_frames)
    out_trans = np.concatenate([np.arange(out_frames[0]),
                                np.arange(out_frames[1]) - out_frames[1]])

    # Output directory formatting.
    if output_dir is not None:
        if output_dir[-1] != '/':
            output_dir += '/'

    datafiles = np.atleast_1d(datafiles)
    # If outlier maps are passed, ensure that there is one for each segment.
    if outlier_maps is not None:
        outlier_maps = np.atleast_1d(outlier_maps)
        if len(outlier_maps) == 1:
            outlier_maps = [outlier_maps[0] for d in datafiles]

    data, fileroots = [], []
    # Load in datamodels from all segments.
    for i, file in enumerate(datafiles):
        currentfile = utils.open_filetype(file)
        data.append(currentfile)
        # Get file name root.
        filename_split = currentfile.meta.filename.split('/')[-1].split('_')
        fileroot = ''
        for seg, segment in enumerate(filename_split):
            if seg == len(filename_split) - 1:
                break
            segment += '_'
            fileroot += segment
        fileroots.append(fileroot)
        # To create the deepstack, join all segments together.
        if i == 0:
            cube = currentfile.data
        else:
            cube = np.concatenate([cube, currentfile.data], axis=0)

    # Generate the deep stack and rms of it. Both 3D (ngroup, dimy, dimx).
    print('Generating a deep stack for each frame using all integrations...')
    deepstack = np.nanmedian(cube[out_trans], axis=0)

    # Individually treat each segment.
    corrected_rampmodels = []
    for n, datamodel in enumerate(data):
        print('Starting segment {} of {}.'.format(n + 1, len(data)))

        # Define the readout setup.
        nint, dimy, dimx = np.shape(datamodel.data)
        # get data quality flags if requested.
        if use_dq is True:
            print(' Considering data quality flags.')
            dq = datamodel.dq
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

        # Read in the main trace mask - a (dimy, dimx) or (2, dimy, dimx)
        # data frame.
        if trace_mask is None:
            print(' No trace mask passed, ignoring the trace.')
            tracemask1 = np.zeros((3, dimy, dimx))
        else:
            print(' Masking the trace.')
            if isinstance(trace_mask, str):
                tracemask1 = fits.getdata(trace_mask)
            elif isinstance(trace_mask, np.ndarray):
                tracemask1 = trace_mask
            else:
                msg = 'Unrecognized trace_mask file type: {}.' \
                      'Ignoring the trace mask.'.format(type(trace_mask))
                warnings.warn(msg)
                tracemask1 = np.zeros((3, dimy, dimx))
        # Trace mask may be order specific, or all order combined. Collapse
        # into a combined mask.
        if np.ndim(tracemask1) == 3:
            tracemask = tracemask1[0].astype(bool) | \
                tracemask1[1].astype(bool) | tracemask1[2].astype(bool)
        else:
            tracemask = tracemask1
        # Convert into a multiplicative mask of 1s and NaNs.
        tracemask = np.where(tracemask == 0, 1, np.nan)
        # Reshape into (nints, dimy, dimx) format.
        tracemask = np.repeat(tracemask, nint).reshape((dimy, dimx, nint))
        tracemask = tracemask.transpose(2, 0, 1)
        # Combine the two main masks.
        outliers = (outliers + tracemask) // 2

        # Since 1/f noise can also vary along a column, there is an option to
        # subtract a median evaluated in a window around the trace for each
        # order.
        if trace_mask2 is not None:
            window = True
            print(' Also subtracting a window median around the trace')
            if isinstance(trace_mask2, str):
                tracemask2 = fits.getdata(trace_mask2)
            elif isinstance(trace_mask2, np.ndarray):
                tracemask2 = trace_mask2
            else:
                msg = 'Unrecognized trace_mask2 file type: {}.' \
                      'Ignoring the trace mask.'.format(type(trace_mask2))
                warnings.warn(msg)
                window = False
        else:
            window = False

        # Initialize output storage arrays.
        dcmap = np.copy(datamodel.data)
        dcmap_w = np.zeros((2, nint, dimy, dimx))
        sub, sub_m = np.copy(datamodel.data), np.copy(datamodel.data)
        subcorr, subcorr_w = np.copy(datamodel.data), np.copy(datamodel.data)
        # Loop over all integrations to determine the 1/f noise level via a
        # difference image, and correct it.
        for i in tqdm(range(nint)):
            # Create two difference images; one to be masked and one not.
            sub[i] = datamodel.data[i] - deepstack*scaling[i]
            sub_m[i] = datamodel.data[i] - deepstack*scaling[i]
            # Since the variable upon which 1/f noise depends is time, treat
            # each group individually.

            # Consider the DQ mask for the group.
            current_outlier = (outliers[i, :, :] + dq[i, :, :]) // 2
            # Apply the outlier mask.
            sub_m[i, :, :] *= current_outlier
            # FULL frame uses multiple amplifiers and probably has to be
            # treated differently.
            if datamodel.meta.subarray.name == 'FULL':
                raise NotImplementedError
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                dc = np.nanmedian(sub_m[i], axis=0)
            # dc is 1D (dimx) - expand to 2D (dimy, dimx)
            dc2d = np.repeat(dc, 256).reshape((2048, 256)).transpose(1, 0)
            # Save the noise map
            dcmap[i, :, :] = dc2d
            # Subtract the noise map to create a corrected difference
            # image - mostly for visualization purposes.
            subcorr[i, :, :] = sub[i, :, :] - dcmap[i, :, :]
            # If a median is also to be subtracted aroud the trace for
            # each order, that is done now.
            if window is True:
                # Treat each order individually.
                for order in [0, 1]:
                    # Create the "window" as the difference between the
                    # two trace masks.
                    trace_window = tracemask2[order] - tracemask1[order]
                    # Convert to multiplicative 1s and NaNs mask.
                    trace_window = np.where(trace_window == 0, 1, np.nan)
                    # Apply the "window" mask.
                    windowed = subcorr[i, :, :] * trace_window
                    # Calculate the DC noise level.
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore',
                                              category=RuntimeWarning)
                        dc_w = np.nanmedian(windowed, axis=0)
                        # Again, noise level is 1D - extend to 2D.
                    dc2d_w = np.repeat(dc_w, 256).reshape((2048, 256))
                    dc2d_w = dc2d_w.transpose(1, 0)
                    # Save the noise level and create a new corrected
                    # difference image.
                    dcmap_w[order, i, :, :] = dc2d_w * tracemask2[order]
                    subcorr_w[i, :, :] = subcorr[i, :, :] - dcmap_w[order, i, :, :]

        # Make sure no NaNs are in the DC map
        dcmap = np.where(np.isfinite(dcmap), dcmap, 0)

        # Subtract the DC map from a copy of the data model
        rampmodel_corr = datamodel.copy()
        corr_data = datamodel.data - dcmap
        # If the window median was also calculated, subtract this noise level
        # as well.
        if window is True:
            corr_data = corr_data - dcmap_w[0] - dcmap_w[1]
            subcorr = np.stack([subcorr, subcorr_w])
            dcmap = np.stack([dcmap, dcmap_w[0], dcmap_w[1]])
        rampmodel_corr.data = corr_data

        # Save the results if requested.
        if save_results is True:
            # Inital difference image.
            hdu = fits.PrimaryHDU(sub)
            hdu.writeto(output_dir + fileroots[n] + 'oneoverfstep_diffim.fits',
                        overwrite=True)
            # 1/f noise-corrected difference image.
            hdu = fits.PrimaryHDU(subcorr)
            hdu.writeto(output_dir + fileroots[n] + 'oneoverfstep_diffimcorr.fits',
                        overwrite=True)
            # DC noise map.
            hdu = fits.PrimaryHDU(dcmap)
            hdu.writeto(output_dir + fileroots[n] + 'oneoverfstep_noisemap.fits',
                        overwrite=True)
            corrected_rampmodels.append(rampmodel_corr)
            # Corrected ramp model.
            rampmodel_corr.write(output_dir + fileroots[n] + 'oneoverfstep.fits')

        # Close datamodel for current segment.
        datamodel.close()

    return corrected_rampmodels


def oneoverfstep(datafiles, out_frames, scaling_curve=None, output_dir=None, save_results=True,
                 outlier_maps=None, trace_mask=None, trace_mask2=None,
                 use_dq=True):
    """Custom 1/f correction routine to be applied at the group level. A
    median stack is constructed using all integrations and subtracted from
    each individual integration. The column-wise median of this difference
    image is then subtracted from the original frame to correct 1/f noise.
    Outlier pixels, as well as the trace itself can be masked to improve the
    noise level estimation. Furthermore, as 1/f noise can also vary with row,
    a column-wise median evaluated in a window around the first and second
    order traces can also be subtracted.

    Parameters
    ----------
    datafiles : list[str], or list[RampModel]
        List of paths to data files, or RampModels themselves for each segment
        of the TSO. Should be 4D ramps and not rate files.
    output_dir : str, None
        Directory to which to save results.
    save_results : bool
        If True, save results to disk.
    outlier_maps : list[str], None
        List of paths to outlier maps for each data segment. Can be
        3D (nints, dimy, dimx), or 2D (dimy, dimx) files.
    trace_mask : str, np.array, None
        Trace mask, or path to file containing a trace mask. Should be 3D
        (norder, dimy, dimx), or 2D (dimy, dimx), although if a "window"
        median subtraction is to be performed, a 3D file is required.
    trace_mask2 : str, np.array, None
        Trace mask, or path to file containing a trace mask. Should be 3D
        (norder, dimy, dimx). If provided, a median evaluated between
        trace_mask and trace_mask2 will also be subtracted.
    use_dq : bool
        If True, mask all pixels currently flagged in the DQ array.

    Returns
    -------
    corrected_rampmodels : list
        RampModels for each segment, corrected for 1/f noise.
    """

    print('Starting custom 1/f correction step.')

    # Output directory formatting.
    if output_dir is not None:
        if output_dir[-1] != '/':
            output_dir += '/'

    out_frames = np.abs(out_frames)
    out_trans = np.concatenate([np.arange(out_frames[0]),
                                np.arange(out_frames[1]) - out_frames[1]])

    datafiles = np.atleast_1d(datafiles)
    # If outlier maps are passed, ensure that there is one for each segment.
    if outlier_maps is not None:
        outlier_maps = np.atleast_1d(outlier_maps)
        if len(outlier_maps) == 1:
            outlier_maps = [outlier_maps[0] for d in datafiles]

    data, fileroots = [], []
    # Load in datamodels from all segments.
    for i, file in enumerate(datafiles):
        currentfile = utils.open_filetype(file)
        data.append(currentfile)
        # Get file name root.
        filename_split = currentfile.meta.filename.split('/')[-1].split('_')
        fileroot = ''
        for seg, segment in enumerate(filename_split):
            if seg == len(filename_split) - 1:
                break
            segment += '_'
            fileroot += segment
        fileroots.append(fileroot)
        # To create the deepstack, join all segments together.
        if i == 0:
            cube = currentfile.data
        else:
            cube = np.concatenate([cube, currentfile.data], axis=0)

    # Generate the deep stack and rms of it. Both 3D (ngroup, dimy, dimx).
    print('Generating a deep stack for each frame using all integrations...')
    deepstack = utils.make_deepstack(cube[out_trans])

    # Make scaling factors
    if scaling_curve is None:
        postage = cube[:, -1, 20:60, 1500:1550]
        timeseries = np.sum(postage, axis=(1, 2))
        scaling_curve = timeseries / np.median(timeseries[0:80])

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

        # Read in the main trace mask - a (dimy, dimx) or (2, dimy, dimx)
        # data frame.
        if trace_mask is None:
            print(' No trace mask passed, ignoring the trace.')
            tracemask1 = np.zeros((3, dimy, dimx))
        else:
            print(' Masking the trace.')
            if isinstance(trace_mask, str):
                tracemask1 = fits.getdata(trace_mask)
            elif isinstance(trace_mask, np.ndarray):
                tracemask1 = trace_mask
            else:
                msg = 'Unrecognized trace_mask file type: {}.' \
                      'Ignoring the trace mask.'.format(type(trace_mask))
                warnings.warn(msg)
                tracemask1 = np.zeros((3, dimy, dimx))
        # Trace mask may be order specific, or all order combined. Collapse
        # into a combined mask.
        if np.ndim(tracemask1) == 3:
            tracemask = tracemask1[0].astype(bool) | \
                tracemask1[1].astype(bool) | tracemask1[2].astype(bool)
        else:
            tracemask = tracemask1
        # Convert into a multiplicative mask of 1s and NaNs.
        tracemask = np.where(tracemask == 0, 1, np.nan)
        # Reshape into (nints, dimy, dimx) format.
        tracemask = np.repeat(tracemask, nint).reshape((dimy, dimx, nint))
        tracemask = tracemask.transpose(2, 0, 1)
        # Combine the two main masks.
        outliers = (outliers + tracemask) // 2

        # Since 1/f noise can also vary along a column, there is an option to
        # subtract a median evaluated in a window around the trace for each
        # order.
        if trace_mask2 is not None:
            window = True
            print(' Also subtracting a window median around the trace')
            if isinstance(trace_mask2, str):
                tracemask2 = fits.getdata(trace_mask2)
            elif isinstance(trace_mask2, np.ndarray):
                tracemask2 = trace_mask2
            else:
                msg = 'Unrecognized trace_mask2 file type: {}.' \
                      'Ignoring the trace mask.'.format(type(trace_mask2))
                warnings.warn(msg)
                window = False
        else:
            window = False

        # Initialize output storage arrays.
        dcmap = np.copy(datamodel.data)
        dcmap_w = np.zeros((2, nint, ngroup, dimy, dimx))
        sub, sub_m = np.zeros_like(datamodel.data), np.zeros_like(datamodel.data)
        subcorr, subcorr_w = np.zeros_like(datamodel.data), np.zeros_like(datamodel.data)
        # Loop over all integrations to determine the 1/f noise level via a
        # difference image, and correct it.
        for i in tqdm(range(nint)):
            # Create two difference images; one to be masked and one not.
            ii = current_int + i
            sub[i] = datamodel.data[i] - deepstack * scaling_curve[ii]
            sub_m[i] = datamodel.data[i] - deepstack * scaling_curve[ii]
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
                dc2d = np.repeat(dc, 256).reshape((2048, 256)).transpose(1, 0)
                # Save the noise map
                dcmap[i, g, :, :] = dc2d
                # Subtract the noise map to create a corrected difference
                # image - mostly for visualization purposes.
                subcorr[i, g, :, :] = sub[i, g, :, :] - dcmap[i, g, :, :]
                # If a median is also to be subtracted aroud the trace for
                # each order, that is done now.
                if window is True:
                    # Treat each order individually.
                    for order in [0, 1]:
                        # Create the "window" as the difference between the
                        # two trace masks.
                        trace_window = tracemask2[order] - tracemask1[order]
                        # Convert to multiplicative 1s and NaNs mask.
                        trace_window = np.where(trace_window == 0, 1, np.nan)
                        # Apply the "window" mask.
                        windowed = subcorr[i, g, :, :] * trace_window
                        # Calculate the DC noise level.
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore',
                                                  category=RuntimeWarning)
                            dc_w = np.nanmedian(windowed, axis=0)
                            # Again, noise level is 1D - extend to 2D.
                        dc2d_w = np.repeat(dc_w, 256).reshape((2048, 256))
                        dc2d_w = dc2d_w.transpose(1, 0)
                        # Save the noise level and create a new corrected
                        # difference image.
                        dcmap_w[order, i, g, :, :] = dc2d_w * tracemask2[order]
                        subcorr_w[i, g, :, :] = subcorr[i, g, :, :] - dcmap_w[order, i, g, :, :]
        current_int += nint

        # Make sure no NaNs are in the DC map
        dcmap = np.where(np.isfinite(dcmap), dcmap, 0)

        # Subtract the DC map from a copy of the data model
        rampmodel_corr = datamodel.copy()
        corr_data = datamodel.data - dcmap
        # If the window median was also calculated, subtract this noise level
        # as well.
        if window is True:
            corr_data = corr_data - dcmap_w[0] - dcmap_w[1]
            subcorr = np.stack([subcorr, subcorr_w])
            dcmap = np.stack([dcmap, dcmap_w[0], dcmap_w[1]])
        rampmodel_corr.data = corr_data

        # Save the results if requested.
        if save_results is True:
            # Inital difference image.
            hdu = fits.PrimaryHDU(sub)
            hdu.writeto(output_dir + fileroots[n] + 'oneoverfstep_diffim.fits',
                        overwrite=True)
            # 1/f noise-corrected difference image.
            hdu = fits.PrimaryHDU(subcorr)
            hdu.writeto(output_dir + fileroots[n] + 'oneoverfstep_diffimcorr.fits',
                        overwrite=True)
            # DC noise map.
            hdu = fits.PrimaryHDU(dcmap)
            hdu.writeto(output_dir + fileroots[n] + 'oneoverfstep_noisemap.fits',
                        overwrite=True)
            corrected_rampmodels.append(rampmodel_corr)
            # Corrected ramp model.
            rampmodel_corr.write(output_dir + fileroots[n] + 'oneoverfstep.fits')

        # Close datamodel for current segment.
        datamodel.close()

    return corrected_rampmodels


def run_stage1(results, background_model, out_frames, scaling_curve=None, save_results=True, outlier_maps=None, trace_mask=None,
               trace_mask2=None, force_redo=False, rejection_threshold=5, show_plots=False,
               root_dir='./', output_tag=''):
    """Run the supreme-SPOON Stage 1 pipeline: detector level processing,
    using a combination of official STScI DMS and custom steps. Documentation
    for the official DMS steps can be found here:
    https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_detector1.html

    Parameters
    ----------
    results : list[str]
        List of paths to input uncalibrated datafiles for all segments in an
        exposure.
    save_results : bool
        If True, save resukts of each step to file.
    outlier_maps : list[str], None
        For improved 1/f noise corecton. List of paths to outlier maps for each
        data segment. Can be 3D (nints, dimy, dimx), or 2D (dimy, dimx) files.
    trace_mask : str, np.array, None
        For improved 1/f noise correcton. Trace mask, or path to file
        containing a trace mask. Should be 3D (norder, dimy, dimx), or 2D
        (dimy, dimx), although if a "window" median subtraction is to be
        performed, a 3D file is required.
    trace_mask2 : str, np.array, None
        For improved 1/f noise correcton. Trace mask, or path to file
        containing a trace mask. Should be 3D  (norder, dimy, dimx). If
        provided, a median evaluated between trace_mask and trace_mask2 will
        also be subtracted.
    force_redo : bool
        If True, redo steps even if outputs files are already present.
    rejection_threshold : int
        For jump detection; sigma threshold for a pixel to be considered an
        outlier.
    root_dir : str
        Directory from which all relative paths are defined.
    output_tag : str
        Name tag to append to pipeline outputs directory.

    Returns
    -------
    results : list[str], list[RampModel]
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

    # Get all files curretly n the output directory to check if steps have
    # been completed.
    all_files = glob.glob(outdir + '*')
    results = np.atleast_1d(results)
    # Get file name root.
    fileroots = []
    for file in results:
        if isinstance(file, str):
            data = datamodels.open(file)
        else:
            data = file
        filename_split = data.meta.filename.split('_')
        fileroot = ''
        for chunk in filename_split[:-1]:
            fileroot += chunk + '_'
        fileroots.append(fileroot)

    # ===== Group Scale Step =====
    # Default DMS step.
    step_tag = 'groupscalestep.fits'
    new_results = []
    for i, segment in enumerate(results):
        # If an output file for this segment already exists, skip the step.
        expected_file = outdir + fileroots[i] + step_tag
        if expected_file in all_files and force_redo is False:
            print('Output file {} already exists.'.format(expected_file))
            print('Skipping Group Scale Step.\n')
            res = expected_file
        # If no output files are detected, run the step.
        else:
            step = calwebb_detector1.group_scale_step.GroupScaleStep()
            res = step.call(segment, output_dir=outdir,
                            save_results=save_results)
        new_results.append(res)
    results = new_results

    # ===== Data Quality Initialization Step =====
    # Default DMS step.
    step_tag = 'dqinitstep.fits'
    new_results = []
    for i, segment in enumerate(results):
        # If an output file for this segment already exists, skip the step.
        expected_file = outdir + fileroots[i] + step_tag
        if expected_file in all_files and force_redo is False:
            print('Output file {} already exists.'.format(expected_file))
            print('Skipping Data Quality Initialization Step.\n')
            res = expected_file
        # If no output files are detected, run the step.
        else:
            step = calwebb_detector1.dq_init_step.DQInitStep()
            res = step.call(segment, output_dir=outdir,
                            save_results=save_results)
        new_results.append(res)
    results = new_results

    # ===== Saturation Detection Step =====
    # Default DMS step.
    step_tag = 'saturationstep.fits'
    new_results = []
    for i, segment in enumerate(results):
        # If an output file for this segment already exists, skip the step.
        expected_file = outdir + fileroots[i] + step_tag
        if expected_file in all_files and force_redo is False:
            print('Output file {} already exists.'.format(expected_file))
            print('Skipping Saturation Detection Step.\n')
            res = expected_file
        # If no output files are detected, run the step.
        else:
            step = calwebb_detector1.saturation_step.SaturationStep()
            res = step.call(segment, output_dir=outdir,
                            save_results=save_results)
        new_results.append(res)
    results = new_results

    # ===== Superbias Subtraction Step =====
    # Default DMS step.
    step_tag = 'superbiasstep.fits'
    new_results = []
    for i, segment in enumerate(results):
        # If an output file for this segment already exists, skip the step.
        expected_file = outdir + fileroots[i] + step_tag
        if expected_file in all_files and force_redo is False:
            print('Output file {} already exists.'.format(expected_file))
            print('Skipping Superbias Subtraction Step.\n')
            res = expected_file
        # If no output files are detected, run the step.
        else:
            step = calwebb_detector1.superbias_step.SuperBiasStep()
            res = step.call(segment, output_dir=outdir,
                            save_results=save_results)
        new_results.append(res)
    results = new_results

    # ===== Reference Pixel Correction Step =====
    # Default DMS step.
    step_tag = 'refpixstep.fits'
    new_results = []
    for i, segment in enumerate(results):
        # If an output file for this segment already exists, skip the step.
        expected_file = outdir + fileroots[i] + step_tag
        if expected_file in all_files and force_redo is False:
            print('Output file {} already exists.'.format(expected_file))
            print('Skipping Reference Pixel Correction Step.\n')
            res = expected_file
        # If no output files are detected, run the step.
        else:
            step = calwebb_detector1.refpix_step.RefPixStep()
            res = step.call(segment, output_dir=outdir,
                            save_results=save_results)
        new_results.append(res)
    results = new_results

    # ===== Background Subtraction Step =====
    # Custom DMS step.
    step_tag = 'backgroundstep.fits'
    do_step = 1
    new_results = []
    for i in range(len(results)):
        # If an output file for this segment already exists, skip the step.
        expected_file = outdir + fileroots[i] + step_tag
        if expected_file not in all_files:
            do_step *= 0
        else:
            new_results.append(expected_file)
    if do_step == 1 and force_redo is False:
        print('Output files already exist.')
        print('Skipping Background Subtraction Step.')
        results = new_results
    # If no output files are detected, run the step.
    else:
        if background_model is None:
            msg = 'No background model provided'
            raise ValueError(msg)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            results = backgroundstep(results, background_model,
                                     output_dir=outdir,
                                     save_results=save_results,
                                     show_plots=show_plots)

    # ===== 1/f Noise Correction Step =====
    # Custom DMS step.
    step_tag = 'oneoverfstep.fits'
    do_step = 1
    new_results = []
    for i in range(len(results)):
        # If an output file for this segment already exists, skip the step.
        expected_file = outdir + fileroots[i] + step_tag
        if expected_file not in all_files:
            do_step *= 0
        else:
            new_results.append(outdir + expected_file)
    if do_step == 1 and force_redo is False:
        print('Output files already exist.')
        print('Skipping 1/f Correction Step.\n')
        results = new_results
    # If no output files are detected, run the step.
    else:
        results = oneoverfstep(results, out_frames=out_frames,
                               scaling_curve=scaling_curve, output_dir=outdir,
                               save_results=save_results,
                               outlier_maps=outlier_maps,
                               trace_mask=trace_mask, trace_mask2=trace_mask2)

    # ===== Linearity Correction Step =====
    # Default DMS step.
    step_tag = 'linearitystep.fits'
    new_results = []
    for i, segment in enumerate(results):
        # If an output file for this segment already exists, skip the step.
        expected_file = outdir + fileroots[i] + step_tag
        if expected_file in all_files and force_redo is False:
            print('Output file {} already exists.'.format(expected_file))
            print('Skipping Linearity Correction Step.\n')
            res = expected_file
        # If no output files are detected, run the step.
        else:
            step = calwebb_detector1.linearity_step.LinearityStep()
            res = step.call(segment, output_dir=outdir,
                            save_results=save_results)
            # Hack to remove oneoverfstep tag from file name.
            res = utils.fix_filenames(res, '_oneoverfstep_', outdir)[0]
        new_results.append(res)
    results = new_results

    # ===== Jump Detection Step =====
    # Default DMS step.
    step_tag = 'jump.fits'
    new_results = []
    for i, segment in enumerate(results):
        # If an output file for this segment already exists, skip the step.
        expected_file = outdir + fileroots[i] + step_tag
        if expected_file in all_files and force_redo is False:
            print('Output file {} already exists.'.format(expected_file))
            print('Skipping Jump Detection Step.\n')
            res = expected_file
        # If no output files are detected, run the step.
        else:
            step = calwebb_detector1.jump_step.JumpStep()
            res = step.call(segment, maximum_cores='quarter',
                            rejection_threshold=rejection_threshold,
                            output_dir=outdir, save_results=save_results)
        new_results.append(res)
    results = new_results

    # ===== Ramp Fit Step =====
    # Default DMS step.
    step_tag = 'rampfitstep.fits'
    new_results = []
    for i, segment in enumerate(results):
        # If an output file for this segment already exists, skip the step.
        expected_file = outdir + fileroots[i] + step_tag
        if expected_file in all_files and force_redo is False:
            print('Output file {} already exists.'.format(expected_file))
            print('Skipping Ramp Fit Step.\n')
            res = expected_file
        # If no output files are detected, run the step.
        else:
            step = calwebb_detector1.ramp_fit_step.RampFitStep()
            res = step.call(segment, output_dir=outdir,
                            save_results=save_results)[1]
            # Store pixel flags in seperate files for potential use in 1/f
            # noise correction.
            hdu = fits.PrimaryHDU(res.dq)
            outfile = outdir + fileroots[i] + 'dqpixelflags.fits'
            hdu.writeto(outfile, overwrite=True)
            # Hack to remove _1_ tag from file name.
            res = utils.fix_filenames(res, '_1_', outdir)[0]
        new_results.append(res)
    results = new_results

    # ===== Gain Scale Correcton Step =====
    # Default DMS step.
    step_tag = 'gainscalestep.fits'
    new_results = []
    for i, segment in enumerate(results):
        # If an output file for this segment already exists, skip the step.
        expected_file = outdir + fileroots[i] + step_tag
        if expected_file in all_files and force_redo is False:
            print('Output file {} already exists.'.format(expected_file))
            print('Skipping Gain Scale Correction Step.\n')
            res = expected_file
        # If no output files are detected, run the step.
        else:
            step = calwebb_detector1.gain_scale_step.GainScaleStep()
            res = step.call(segment, output_dir=outdir,
                            save_results=save_results)
        new_results.append(res)
    results = new_results

    return results


if __name__ == "__main__":
    # =============== User Input ===============
    root_dir = './'  # Root directory.
    indir = root_dir + 'DMS_uncal/'  # Uncalibrated file directory.
    input_filetag = 'uncal'  # Unclaibrated file tag.
    background_file = root_dir + 'model_background256.npy'  # Background model.
    outlier_maps = None  # For 1/f correction; outlier pixel maps.
    trace_mask = None  # For 1/f correcton; trace mask.
    trace_mask2 = None  # For 1/f correcton; trace mask for window subtraction.
    exposure_type = 'CLEAR'  # Either CLEAR or F277W.
    force_redo = False  # Force redo of completed steps.
    # ==========================================

    # Set the CRDS cache variables.
    import os
    os.environ['CRDS_PATH'] = root_dir + 'crds_cache'
    os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

    # Unpack all files in the input directory.
    input_files = utils.unpack_input_directory(indir, filetag=input_filetag,
                                               exposure_type=exposure_type)
    print('\nIdentified {0} {1} exposure segments'.format(len(input_files), exposure_type))
    for file in input_files:
        print(' ' + file)

    # Unpack background model.
    background_model = np.load(background_file)

    # Run segments through Stage 1.
    stage1_results = run_stage1(input_files, background_model=background_model,
                                save_results=True,
                                outlier_maps=outlier_maps,
                                trace_mask=trace_mask, trace_mask2=trace_mask2,
                                force_redo=force_redo, root_dir=root_dir)
