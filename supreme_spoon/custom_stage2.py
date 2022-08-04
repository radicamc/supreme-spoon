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
from tqdm import tqdm
import warnings

from jwst import datamodels
from jwst.extract_1d.soss_extract import soss_boxextract
from jwst.pipeline import calwebb_spec2

from supreme_spoon import plotting, utils


def badpixstep(datafiles, thresh=3, box_size=5, max_iter=2, output_dir=None,
               save_results=True):
    """Identify and correct bad pixels remaining in the dataset. Find outlier
    pixels in the median stack and correct them via the median of a box of
    surrounding pixels in each integration.

    Parameters
    ----------
    datafiles : list[str], list[CubeModel]
        List of paths to datafiles for each segment, or the datamodels
        themselves.
    thresh : int
        Sigma threshold for a deviant pixel to be flagged.
    box_size : int
        Size of box around each pixel to test for deviations.
    max_iter : int
        Maximum number of outlier flagging iterations.
    output_dir : str, None
        Directory to which to output results.
    save_results : bool
        If True, save results to file.

    Returns
    -------
    data : list[CubeModel]
        Input datamodels for each segment, corrected for outlier pixels.
    badpix_mask : np.array
        Mask of all pixels flagged by the outlier routine.
    deepframe : np.array
        Final median stack of all outlier corrected integrations.
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
        currentfile = utils.open_filetype(file)
        data.append(currentfile)
        # Hack to get filename root.
        filename = currentfile.meta.filename
        filename_split = filename.split('/')[-1].split('_')
        fileroot = ''
        for seg, segment in enumerate(filename_split):
            if seg == len(filename_split) - 1:
                break
            segment += '_'
            fileroot += segment
        fileroots.append(fileroot)

        # To create the deepstack, join all segments together.
        # Also stack all the dq arrays from each segement.
        if i == 0:
            cube = currentfile.data
            dq_cube = currentfile.dq
        else:
            cube = np.concatenate([cube, currentfile.data], axis=0)
            dq_cube = np.concatenate([dq_cube, currentfile.dq], axis=0)

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
    newdq = np.copy(dq_cube)
    it = 0

    while it < max_iter:
        print('Starting iteration {0} of {1}.'.format(it + 1, max_iter))

        # Generate the deepstack.
        print(' Generating a deep stack using all integrations...')
        deepframe = utils.make_deepstack(newdata)
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
            newdata[itg], newdq[itg] = utils.do_replacement(newdata[itg],
                                                            to_replace,
                                                            dq=newdq[itg],
                                                            box_size=box_size)
        it += 1

    # Ensure that the bad pixel mask remains zeros or ones.
    badpix_mask = np.where(badpix_mask == 0, 0, 1)
    # Generate a final corrected deep frame.
    deepframe = utils.make_deepstack(newdata)

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

    if save_results is True:
        # Save bad pixel mask.
        hdu = fits.PrimaryHDU(badpix_mask)
        hdu.writeto(output_dir + fileroot_noseg + 'badpixmap.fits',
                    overwrite=True)

        # Save deep frame.
        hdu = fits.PrimaryHDU(deepframe)
        hdu.writeto(output_dir + fileroot_noseg + 'deepframe.fits',
                    overwrite=True)

    return data, badpix_mask, deepframe


# TODO: needs to work for str, datamodel or np.array
def tracemaskstep(deepframe, result, output_dir=None, mask_width=30,
                  save_results=True, show_plots=False):
    """Create a mask of a user-specified width around each of the SOSS
    diffraction orders.
    Note that the centroiding algorithm requires the APPLESOSS module, and
    will fail if it is not available.

    Parameters
    ----------
    deepframe : str, np.array, RampModel
        Path to median stack file, or the median stack itself. Should be 2D
        (dimy, dimx).
    output_dir : str, None
        Directory to which to save outputs.
    mask_width : int
        Mask width, in pixels, around the trace centroids.
    save_results : bool
        If Tre, save results to file.
    show_plots : bool
        If True, display plots.

    Returns
    -------
    tracemask : np.array
        3D (norder, dimy, dimx) trace mask.
    """

    # get the median stack data.
    # deepframe = utils.open_filetype(deepframe)
    # deepframe = deepframe.extra_fits.PRIMARY.data

    result = utils.open_filetype(result)
    working_name = result.meta.filename
    # Get total file root, with no segment info.
    if 'seg' in working_name:
        parts = working_name.split('seg')
        part1, part2 = parts[0][:-1], parts[1][3:8]
        fileroot = part1 + part2
    else:
        fileroot = working_name

    # Get centroids for orders one to three.
    dimy, dimx = np.shape(deepframe)
    if dimy == 256:
        subarray = 'SUBSTRIP256'
    else:
        raise NotImplementedError
    # Get centroids via the edgetrigger method
    centroids = utils.get_trace_centroids(deepframe, subarray,
                                          save_results=False)
    x1, y1 = centroids[0][0], centroids[0][1]
    x2, y2 = centroids[1][0], centroids[1][1]
    x3, y3 = centroids[2][0], centroids[2][1]
    # Show the extracted centroids over the deepframe is requested.
    if show_plots is True:
        plotting.do_centroid_plot(deepframe, x1, y1, x2, y2, x3, y3)

    # Create the masks for each order.
    weights1 = soss_boxextract.get_box_weights(y1, mask_width, (dimy, dimx),
                                               cols=x1.astype(int))
    weights1 = np.where(weights1 == 0, 0, 1)
    weights2 = soss_boxextract.get_box_weights(y2, mask_width, (dimy, dimx),
                                               cols=x2.astype(int))
    weights2 = np.where(weights2 == 0, 0, 1)
    weights3 = soss_boxextract.get_box_weights(y3, mask_width, (dimy, dimx),
                                               cols=x3.astype(int))
    weights3 = np.where(weights3 == 0, 0, 1)

    # Pack the masks into an array.
    tracemask = np.zeros((3, dimy, dimx))
    tracemask[0] = weights1
    tracemask[1] = weights2
    tracemask[2] = weights3
    # Plot the mask if requested.
    if show_plots is True:
        plotting.do_tracemask_plot(weights1 | weights2 | weights3)

    # Save the trace mask to file if requested.
    if save_results is True:
        hdu = fits.PrimaryHDU(tracemask)
        hdu.writeto(output_dir + fileroot + 'tracemask_width{}.fits'.format(mask_width),
                    overwrite=True)

    return tracemask


def lcestimatestep(datafiles, out_frames, save_results=True, output_dir=None):
    out_frames = np.abs(out_frames)
    out_trans = np.concatenate([np.arange(out_frames[0]),
                                np.arange(out_frames[1]) - out_frames[1]])

    datafiles = np.atleast_1d(datafiles)

    for i, file in enumerate(datafiles):
        current_data = utils.open_filetype(file)
        if i == 0:
            cube = current_data.data
        else:
            cube = np.concatenate([cube, current_data.data], axis=0)

    filename = file.meta.filename.split('/')[-1]
    parts = filename.split('seg')
    part1, part2 = parts[0][:-1], parts[1][3:]
    fileroot_noseg = part1 + part2

    postage = cube[:, 20:60, 1500:1550]
    timeseries = np.sum(postage, axis=(1, 2))
    curve = timeseries / np.median(timeseries[out_trans])

    if save_results is True:
        outfile = output_dir + fileroot_noseg + 'lcscaling.npy'
        np.save(outfile, curve)

    return curve


def run_stage2(results, out_frames, save_results=True,
               force_redo=False, max_iter=2, mask_width=30,
               root_dir='./', output_tag=''):
    # ============== DMS Stage 2 ==============
    # Spectroscopic processing.
    # Documentation: https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_spec2.html
    print('\n\n**Starting supreme-SPOON Stage 2**')
    print('Spectroscopic processing\n\n')

    if output_tag != '':
        output_tag = '_' + output_tag
    # Create output directories and define output paths.
    utils.verify_path(root_dir + 'pipeline_outputs_directory' + output_tag)
    utils.verify_path(root_dir + 'pipeline_outputs_directory' + output_tag + '/Stage2')
    outdir = root_dir + 'pipeline_outputs_directory' + output_tag + '/Stage2/'

    all_files = glob.glob(outdir + '*')
    results = np.atleast_1d(results)
    # Get file root
    fileroots = []
    for file in results:
        data = utils.open_filetype(file)
        filename_split = data.meta.filename.split('_')
        fileroot = ''
        for chunk in filename_split[:-1]:
            fileroot += chunk + '_'
        fileroots.append(fileroot)

    # ===== Assign WCS Step =====
    # Default DMS step.
    step_tag = 'assignwcsstep.fits'
    new_results = []
    for i, segment in enumerate(results):
        # If an output file for this segment already exists, skip the step.
        expected_file = outdir + fileroots[i] + step_tag
        if expected_file in all_files and force_redo is False:
            print('Output file {} already exists.'.format(expected_file))
            print('Skipping Assign WCS Step.')
            res = expected_file
        # If no output files are detected, run the step.
        else:
            step = calwebb_spec2.assign_wcs_step.AssignWcsStep()
            res = step.call(segment, output_dir=outdir,
                            save_results=save_results)
        new_results.append(res)
    results = new_results

    # ===== Source Type Determination Step =====
    # Default DMS step.
    step_tag = 'sourcetypestep.fits'
    new_results = []
    for i, segment in enumerate(results):
        # If an output file for this segment already exists, skip the step.
        expected_file = outdir + fileroots[i] + step_tag
        if expected_file in all_files and force_redo is False:
            print('Output file {} already exists.'.format(expected_file))
            print('Skipping Source Type Determination Step.')
            res = expected_file
        # If no output files are detected, run the step.
        else:
            step = calwebb_spec2.srctype_step.SourceTypeStep()
            res = step.call(segment, output_dir=outdir,
                            save_results=save_results)
        new_results.append(res)
    results = new_results

    # ===== Flat Field Correction Step =====
    # Default DMS step.
    step_tag = 'flatfieldstep.fits'
    new_results = []
    for i, segment in enumerate(results):
        # If an output file for this segment already exists, skip the step.
        expected_file = outdir + fileroots[i] + step_tag
        if expected_file in all_files and force_redo is False:
            print('Output file {} already exists.'.format(expected_file))
            print('Skipping Flat Field Correction Step.')
            res = expected_file
        # If no output files are detected, run the step.
        else:
            step = calwebb_spec2.flat_field_step.FlatFieldStep()
            res = step.call(segment, output_dir=outdir,
                            save_results=save_results)
        new_results.append(res)
    results = new_results

    # ===== Bad Pixel Correction Step =====
    # Custom DMS step.
    step_tag = 'badpixstep.fits'
    do_step = 1
    new_results = []
    for i in range(len(results)):
        # If an output file for this segment already exists, skip the step.
        expected_file = outdir + fileroots[i] + step_tag
        if expected_file not in all_files:
            do_step *= 0
        else:
            existing_data = datamodels.open(expected_file)
            new_results.append(existing_data)
    if do_step == 1 and force_redo is False:
        print('Output files already exist.')
        print('Skipping Bad Pixel Correction Step.')
        results = new_results
        # Get total file root, with no segment info.
        working_name = fileroots[0]
        if 'seg' in working_name:
            parts = working_name.split('seg')
            part1, part2 = parts[0][:-1], parts[1][3:]
            fileroot_noseg = part1 + part2
        else:
            fileroot_noseg = fileroots[0]
        # Also get the median stack.
        existing_deep = fileroot_noseg + 'deepframe.fits'
        deepframe = fits.getdata(outdir + existing_deep, 0)
    # If no output files are detected, run the step.
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = badpixstep(results, output_dir=outdir,
                             save_results=save_results, max_iter=max_iter)
            results, deepframe = res[0], res[2]

    # ===== Trace Mask Creation Step =====
    # Custom DMS step.
    step_tag = 'tracemask_width{}.fits'.format(mask_width)
    # If an output file for this segment already exists, skip the step.
    fileroot_split = fileroots[0].split('-seg')
    fileroot_noseg = fileroot_split[0] + fileroot_split[1][3:]
    expected_file = outdir + fileroot_noseg + step_tag
    if expected_file in all_files and force_redo is False:
        print('Output file {} already exists.'.format(expected_file))
        print('Skipping Trace Mask Creation Step.')
        mask = expected_file
    else:
        print('Starting Trace Mask Creation Step.')
        mask = tracemaskstep(deepframe, results[0], output_dir=outdir,
                             mask_width=mask_width, save_results=save_results,
                             show_plots=False)

    # ===== Light Curve Scaling Estimation Step =====
    # Custom DMS step.
    step_tag = 'lcscaling.npy'.format(mask_width)
    # If an output file for this segment already exists, skip the step.
    fileroot_split = fileroots[0].split('-seg')
    fileroot_noseg = fileroot_split[0] + fileroot_split[1][3:]
    expected_file = outdir + fileroot_noseg + step_tag
    if expected_file in all_files and force_redo is False:
        print('Output file {} already exists.'.format(expected_file))
        print('Skipping Light Curve Estimation Step.')
        scaling = expected_file
    else:
        print('Starting Trace Mask Creation Step.')
        scaling = lcestimatestep(results, out_frames=out_frames,
                                 save_results=save_results, output_dir=outdir)

    return results, deepframe
