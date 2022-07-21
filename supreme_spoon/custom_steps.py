#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 08:53 2022

@author: MCR

Custom JWST DMS pipeline steps.
"""

from astropy.io import fits
import numpy as np
from tqdm import tqdm
import warnings

from jwst import datamodels
from jwst.extract_1d.soss_extract import soss_boxextract

from sys import path
applesoss_path = '/home/radica/GitHub/APPLESOSS/'
path.insert(1, applesoss_path)

from APPLESOSS.edgetrigger_centroids import get_soss_centroids

import plotting
import utils


def badpixstep(datafiles, thresh=3, box_size=5, max_iter=2, output_dir=None,
               save_results=True):
    """Interpolate bad pixels flagged in the deep frame in individual
    integrations.
    """

    print('Starting custom outlier interpolation step.')

    # Output directory formatting.
    if output_dir is not None:
        if output_dir[-1] != '/':
            output_dir += '/'

    datafiles = np.atleast_1d(datafiles)

    data, fileroots = [], []
    # Load in datamodels from all segments.
    for i, file in enumerate(datafiles):
        currentfile = datamodels.open(file)
        data.append(currentfile)
        # Hack to get filename root.
        filename_split = file.split('/')[-1].split('_')
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

    # Get total file root, with no segment info.
    working_name = fileroots[0]
    if 'seg' in working_name:
        parts = working_name.split('seg')
        part1, part2 = parts[0][:-1], parts[1][3:]
        fileroot_noseg = part1 + part2
    else:
        fileroot_noseg = fileroots[0]

        # Initialize starting loop variables.
    badpix_mask = np.zeros((256, 2048))
    newdata = np.copy(cube)
    it = 0

    while it < max_iter:
        print('Starting iteration {0} of {1}.'.format(it + 1, max_iter))

        # Generate the deepstack.
        print(' Generating a deep stack using all integrations...')
        deepframe = utils.make_deepstack(newdata)[0]
        badpix = np.zeros_like(deepframe)
        count = 0
        nint, dimy, dimx = np.shape(newdata)

        # On the first iteration only - also interpolate any NaNs in
        # individual integrations.
        if it == 0:
            nanpix = np.isnan(newdata).astype(int)
        else:
            nanpix = np.zeros_like(newdata)

        # Loop over whole deepstack and flag deviant pixels.
        for i in tqdm(range(dimx)):
            for j in range(dimy):
                box_size_i = box_size
                box_prop = utils.get_interp_box(deepframe, box_size_i, i, j,
                                                dimx, dimy)

                # Ensure that the median and std dev extracted are good.
                # If not, increase the box size until they are.
                while np.any(np.isnan(box_prop)):
                    box_size_i += 1
                    box_prop = utils.get_interp_box(deepframe, box_size_i, i,
                                                    j, dimx, dimy)
                med, std = box_prop[0], box_prop[1]

                # If central pixel is too deviant (or nan) flag it.
                if np.abs(deepframe[j, i] - med) > thresh * std or np.isnan(
                        deepframe[j, i]):
                    mini, maxi = np.max([0, i - 1]), np.min([dimx - 1, i + 1])
                    minj, maxj = np.max([0, j - 1]), np.min([dimy - 1, j + 1])
                    badpix[j, i] = 1
                    # Also flag cross around the central pixel.
                    badpix[maxj, i] = 1
                    badpix[minj, i] = 1
                    badpix[j, maxi] = 1
                    badpix[j, mini] = 1
                    count += 1

        print(' {} bad pixels identified this iteration.'.format(count))
        # End if no bad pixels are found.
        if count == 0:
            break
        # Add bad pixels flagged this iteration to total mask.
        badpix_mask += badpix
        # Replace the flagged pixels in each individual integration.
        for itg in tqdm(range(nint)):
            to_replace = badpix + nanpix[itg]
            newdata[itg] = utils.do_replacement(newdata[itg], to_replace,
                                                box_size=box_size)
        it += 1

    # Ensure that the bad pixels mask remains zeros or ones.
    badpix_mask = np.where(badpix_mask == 0, 0, 1)

    if save_results is True:
        current_int = 0
        # Save interpolated data.
        for n, file in enumerate(data):
            currentdata = file.data
            nints = np.shape(currentdata)[0]
            file.data = newdata[current_int:(current_int + nints)]
            file.write(output_dir + fileroots[n] + 'badpixstep.fits')
            current_int += nints

        # Save bad pixel mask.
        hdu = fits.PrimaryHDU(badpix_mask)
        hdu.writeto(output_dir + fileroot_noseg + 'badpixmap.fits',
                    overwrite=True)

    return newdata, badpix_mask


def oneoverfstep(datafiles, output_dir=None, save_results=True,
                 outlier_maps=None, trace_mask=None, use_dq=True):
    """Custom 1/f correction routine to be applied at the group level.

    Parameters
    ----------
    datafiles : list[str], or list[RampModel]
        List of paths to data files, or RampModels themselves for each segment
        of the TSO. Should be 4D ramps and not rate files.
    output_dir : str
        Directory to which to save results.
    save_results : bool
        If True, save results to disk.
    outlier_maps : list[str]
        List of paths to outlier maps for each data segment. Can be
        3D (nints, dimy, dimx), or 2D (dimy, dimx) files.
    trace_mask : str
        Path to trace mask file. Should be 2D (dimy, dimx).
    use_dq : bool
        If True, also mask all pixels flagged in the DQ array.

    Returns
    -------
    corrected_rampmodels : list
        Ramp models for each segment corrected for 1/f noise.
    """

    print('Starting custom 1/f correction step.')

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
        if isinstance(file, str):
            currentfile = datamodels.open(file)
        else:
            currentfile = file
        data.append(currentfile)
        # Hack to get filename root.
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

    # Get total file root, with no segment info.
    working_name = fileroots[0]
    if 'seg' in working_name:
        parts = working_name.split('seg')
        part1, part2 = parts[0][:-1], parts[1][3:]
        fileroot_noseg = part1+part2
    else:
        fileroot_noseg = fileroots[0]

    # Generate the deep stack and rms of it. Both 3D (ngroup, dimy, dimx).
    print('Generating a deep stack for each frame using all integrations...')
    deepstack, rms = utils.make_deepstack(cube, return_rms=True)
    # Save these to disk if requested.
    if save_results is True:
        hdu = fits.PrimaryHDU(deepstack)
        hdu.writeto(output_dir+fileroot_noseg+'oneoverfstep_deepstack.fits',
                    overwrite=True)
        hdu = fits.PrimaryHDU(rms)
        hdu.writeto(output_dir+fileroot_noseg+'oneoverfstep_rms.fits',
                    overwrite=True)

    corrected_rampmodels = []
    for n, datamodel in enumerate(data):
        print('Starting segment {} of {}.'.format(n + 1, len(data)))

        # The readout setup
        ngroup = datamodel.meta.exposure.ngroups
        nint = np.shape(datamodel.data)[0]
        dimx = np.shape(datamodel.data)[-1]
        dimy = np.shape(datamodel.data)[-2]
        # Also open the data quality flags if requested.
        if use_dq is True:
            print(' Considering data quality flags.')
            dq = datamodel.groupdq
            # Mask will be applied multiplicatively.
            dq = np.where(dq == 0, 1, np.nan)
        else:
            dq = np.ones_like(datamodel.data)

        # Weighted average to determine the 1/f DC level
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            w = 1 / rms

        # Read in the outlier map -- a (nints, dimy, dimx) 3D cube
        if outlier_maps is None:
            msg = ' No outlier maps passed, ignoring outliers.'
            print(msg)
            outliers = np.zeros((nint, dimy, dimx))
        else:
            print(' Using outlier map {}'.format(outlier_maps[n]))
            outliers = fits.getdata(outlier_maps[n])
            # If the outlier map is 2D (dimy, dimx) extend to int dimension.
            if np.ndim(outliers) == 2:
                outliers = np.repeat(outliers, nint).reshape(dimy, dimx, nint)
                outliers = outliers.transpose(2, 0, 1)

        # Read in the trace mask -- a (dimy, dimx) data frame.
        if trace_mask is None:
            msg = ' No trace mask passed, ignoring the trace.'
            print(msg)
            tracemask = np.zeros((dimy, dimx))
        else:
            print(' Using trace mask {}'.format(trace_mask))
            tracemask = fits.getdata(trace_mask)

        # The outlier map is 0 where good and >0 otherwise. As this will be
        # applied multiplicatively, replace 0s with 1s and others with NaNs.
        outliers = np.where(outliers == 0, 1, np.nan)
        # Same thing with the trace mask.
        tracemask = np.where(tracemask == 0, 1, np.nan)
        tracemask = np.repeat(tracemask, nint).reshape(dimy, dimx, nint)
        tracemask = tracemask.transpose(2, 0, 1)
        # Combine the two masks.
        outliers = (outliers + tracemask) // 2

        dcmap = np.copy(datamodel.data)
        subcorr = np.copy(datamodel.data)
        sub, sub_m = np.copy(datamodel.data), np.copy(datamodel.data)
        for i in tqdm(range(nint)):
            # Create two difference images; one to be masked and one not.
            sub[i] = datamodel.data[i] - deepstack
            sub_m[i] = datamodel.data[i] - deepstack
            for g in range(ngroup):
                # Add in DQ mask.
                current_outlier = (outliers[i, :, :] + dq[i, g, :, :]) // 2
                # Apply the outlier mask.
                sub_m[i, g, :, :] *= current_outlier
                # Make sure to not subtract an overall bias
                sub[i, g, :, :] -= np.nanmedian(sub[i, g, :, :])
                sub_m[i, g, :, :] -= np.nanmedian(sub[i, g, :, :])
            if datamodel.meta.subarray.name == 'SUBSTRIP256':
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    dc = np.nansum(w * sub_m[i], axis=1)
                    dc /= np.nansum(w * current_outlier, axis=1)
                # make sure no NaN will corrupt the whole column
                dc = np.where(np.isfinite(dc), dc, 0)
                # dc is 2D - expand to the 3rd (columns) dimension
                dc3d = np.repeat(dc, 256).reshape((ngroup, 2048, 256))
                dcmap[i, :, :, :] = dc3d.swapaxes(1, 2)
                subcorr[i, :, :, :] = sub[i, :, :, :] - dcmap[i, :, :, :]
            elif datamodel.meta.subarray.name == 'SUBSTRIP96':
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    dc = np.nansum(w * sub_m[i], axis=1)
                    dc /= np.nansum(w * current_outlier, axis=1)
                # make sure no NaN will corrupt the whole column
                dc = np.where(np.isfinite(dc), dc, 0)
                # dc is 2D - expand to the 3rd (columns) dimension
                dc3d = np.repeat(dc, 256).reshape((ngroup, 2048, 256))
                dcmap[i, :, :, :] = dc3d.swapaxes(1, 2)
                subcorr[i, :, :, :] = sub[i, :, :, :] - dcmap[i, :, :, :]
            elif datamodel.meta.subarray.name == 'FULL':
                for amp in range(4):
                    yo = amp*512
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=RuntimeWarning)
                        dc = np.nansum(w[:, :, yo:yo+512, :] * sub_m[:, :, yo:yo+512, :], axis=1)
                        dc /= (np.nansum(w[:, :, yo:yo+512, :] * current_outlier, axis=1))
                    # make sure no NaN will corrupt the whole column
                    dc = np.where(np.isfinite(dc), dc, 0)
                    # dc is 2D - expand to the 3rd (columns) dimension
                    dc3d = np.repeat(dc, 512).reshape((ngroup, 2048, 512))
                    dcmap[i, :, yo:yo+512, :] = dc3d.swapaxes(1, 2)
                    subcorr[i, :, yo:yo+512, :] = sub[i, :, yo:yo+512, :] - dcmap[i, :, yo:yo+512, :]

        # Make sure no NaNs are in the DC map
        dcmap = np.where(np.isfinite(dcmap), dcmap, 0)

        # Subtract the DC map from a copy of the data model
        rampmodel_corr = datamodel.copy()
        rampmodel_corr.data = datamodel.data - dcmap

        # Save results to disk if requested.
        if save_results is True:
            hdu = fits.PrimaryHDU(sub)
            hdu.writeto(output_dir+fileroots[n]+'oneoverfstep_diffim.fits',
                        overwrite=True)
            hdu = fits.PrimaryHDU(subcorr)
            hdu.writeto(output_dir+fileroots[n]+'oneoverfstep_diffimcorr.fits',
                        overwrite=True)
            hdu = fits.PrimaryHDU(dcmap)
            hdu.writeto(output_dir+fileroots[n]+'oneoverfstep_noisemap.fits',
                        overwrite=True)
            corrected_rampmodels.append(rampmodel_corr)
            rampmodel_corr.write(output_dir+fileroots[n]+'oneoverfstep.fits')

        datamodel.close()

    return corrected_rampmodels


def backgroundstep(datafiles, background_model, subtract_column_median=False,
                   output_dir=None, save_results=True, show_plots=False):
    # Output directory formatting.
    if output_dir is not None:
        if output_dir[-1] != '/':
            output_dir += '/'

    datafiles = np.atleast_1d(datafiles)
    results = []
    for file in datafiles:
        if isinstance(file, str):
            currentfile = datamodels.open(file)
        else:
            currentfile = file

        old_filename = currentfile.meta.filename
        to_remove = old_filename.split('_')[-1]
        fileroot = old_filename.split(to_remove)[0]

        scale_mod = np.nanmedian(background_model[:, :500])
        scale_dat = np.nanmedian(currentfile.data[:, 200:, :500], axis=(1, 2))
        model_scaled = background_model / scale_mod * scale_dat[:, None, None]
        data_backsub = currentfile.data - model_scaled

        if subtract_column_median is True:
            # Placeholder for median subtraction
            pass
        currentfile.data = data_backsub

        if save_results is True:
            hdu = fits.PrimaryHDU(model_scaled)
            hdu.writeto(output_dir + fileroot + 'background.fits',
                        overwrite=True)

            currentfile.write(output_dir + fileroot + 'backgroundstep.fits')

        if show_plots is True:
            plotting.do_backgroundsubtraction_plot(currentfile.data,
                                                   background_model,
                                                   scale_mod, scale_dat)
        results.append(currentfile.data)
        currentfile.close()

    return results


def make_tracemask(datafiles, output_dir, mask_width=30, save_results=True,
                   show_plots=False):

    datafiles = np.atleast_1d(datafiles)

    for i, file in enumerate(datafiles):
        if isinstance(file, str):
            currentfile = datamodels.open(file)
        else:
            currentfile = file

        if i == 0:
            cube = currentfile.data
            fileroot = currentfile.meta.filename.split('_')[0]
        else:
            cube = np.concatenate([cube, currentfile.data], axis=0)
        currentfile.close()

    deepframe = utils.make_deepstack(cube)[0]

    # Get orders 1 to 3 centroids
    dimy, dimx = np.shape(deepframe)
    if dimy == 256:
        subarray = 'SUBSTRIP256'
    else:
        raise NotImplementedError

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        centroids = get_soss_centroids(deepframe, subarray=subarray)

    X1, Y1 = centroids['order 1']['X centroid'], centroids['order 1'][
        'Y centroid']
    X2, Y2 = centroids['order 2']['X centroid'], centroids['order 2'][
        'Y centroid']
    X3, Y3 = centroids['order 3']['X centroid'], centroids['order 3'][
        'Y centroid']
    ii = np.where((X1 >= 0) & (X1 <= dimx - 1))
    ii2 = np.where((X2 >= 0) & (X2 <= dimx - 1) & (Y2 <= dimy - 1))
    ii3 = np.where((X3 >= 0) & (X3 <= dimx - 1) & (Y3 <= dimy - 1))

    # Interpolate onto native pixel grid
    x1 = np.arange(dimx)
    y1 = np.interp(x1, X1[ii], Y1[ii])
    x2 = np.arange(np.max(np.floor(X2[ii2]).astype(int)))
    y2 = np.interp(x2, X2[ii2], Y2[ii2])
    x3 = np.arange(np.max(np.floor(X3[ii3]).astype(int)))
    y3 = np.interp(x3, X3[ii3], Y3[ii3])

    if show_plots is True:
        plotting.do_centroid_plot(deepframe, x1, y1, x2, y2, x3, y3)

    weights1 = soss_boxextract.get_box_weights(y1, mask_width, (dimy, dimx),
                                               cols=x1.astype(int))
    weights2 = soss_boxextract.get_box_weights(y2, mask_width, (dimy, dimx),
                                               cols=x2.astype(int))
    weights3 = soss_boxextract.get_box_weights(y3, mask_width, (dimy, dimx),
                                               cols=x3.astype(int))
    weights1 = np.where(weights1 == 0, 0, 1)
    weights2 = np.where(weights2 == 0, 0, 1)
    weights3 = np.where(weights3 == 0, 0, 1)

    tracemask = weights1 | weights2 | weights3
    if show_plots is True:
        plotting.do_tracemask_plot(tracemask)

    if save_results is True:
        hdu = fits.PrimaryHDU(tracemask)
        hdu.writeto(output_dir + fileroot + '_tracemask.fits', overwrite=True)

    return tracemask
