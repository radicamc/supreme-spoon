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

from supreme_spoon import utils
from supreme_spoon import plotting


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
    # Generate a final corrected deep frame.
    deepframe = utils.make_deepstack(newdata)[0]

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

        # Save deep frame.
        hdu = fits.PrimaryHDU(deepframe)
        hdu.writeto(output_dir + fileroot_noseg + 'deepframe.fits',
                    overwrite=True)

    return newdata, badpix_mask, deepframe


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


def tracemaskstep(datafiles, output_dir, mask_width=30, save_results=True,
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

    cen_o1, cen_o2, cen_o3 = utils.get_trace_centroids(deepframe, subarray)
    x1, y1 = cen_o1
    x2, y2 = cen_o2
    x3, y3 = cen_o3

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

    return deepframe, tracemask


def run_stage2(results, iteration, background_model=None, save_results=True,
               force_redo=False, show_plots=False, max_iter=2, mask_width=30):
    # ============== DMS Stage 2 ==============
    # Spectroscopic processing.
    # Documentation: https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_spec2.html
    utils.verify_path('pipeline_outputs_directory')
    utils.verify_path('pipeline_outputs_directory/Stage2')
    if iteration == 1:
        utils.verify_path('pipeline_outputs_directory/Stage2/FirstPass')
        outdir = 'pipeline_outputs_directory/Stage2/FirstPass/'
    else:
        utils.verify_path('pipeline_outputs_directory/Stage2/SecondPass')
        outdir = 'pipeline_outputs_directory/Stage2/SecondPass/'

    all_files = glob.glob(outdir + '*')
    results = np.atleast_1d(results)
    # Get file root
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

    # ===== Assign WCS Step =====
    # Default DMS step.
    step_tag = 'assignwcsstep.fits'
    new_results = []
    for i, segment in enumerate(results):
        expected_file = outdir + fileroots[i] + step_tag
        if expected_file in all_files and force_redo is False:
            print('Output file {} already exists.'.format(expected_file))
            print('Skipping Assign WCS Step.')
            res = expected_file
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
        expected_file = outdir + fileroots[i] + step_tag
        if expected_file in all_files and force_redo is False:
            print('Output file {} already exists.'.format(expected_file))
            print('Skipping Source Type Determination Step.')
            res = expected_file
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
        expected_file = outdir + fileroots[i] + step_tag
        if expected_file in all_files and force_redo is False:
            print('Output file {} already exists.'.format(expected_file))
            print('Skipping Flat Field Correction Step.')
            res = expected_file
        else:
            step = calwebb_spec2.flat_field_step.FlatFieldStep()
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
        expected_file = outdir + fileroots[i] + step_tag
        if expected_file not in all_files:
            do_step *= 0
        else:
            new_results.append(expected_file)
    if do_step == 1 and force_redo is False:
        print('Output files already exist.')
        print('Skipping Background Subtraction Step.')
        results = new_results
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

    # ===== Trace Mask Creation Step =====
    # Custom DMS step.
    step_tag = 'tracemask.fits'
    abs_root = fileroots[0].split('/')[0]
    expected_file = abs_root + '_' + step_tag
    if expected_file in all_files and force_redo is False:
        print('Output file {} already exists.'.format(expected_file))
        print('Skipping Trace Mask Creation Step.')
        tracemask = expected_file
    else:
        res = tracemaskstep(results, output_dir=outdir,
                            save_results=save_results, show_plots=show_plots,
                            mask_width=mask_width)
        tracemask = res[1]

    # ===== Bad Pixel Correction Step =====
    # Custom DMS step.
    if iteration == 2:
        step_tag = 'badpixstep.fits'
        do_step = 1
        new_results = []
        for i in range(len(results)):
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
            existing_deep = fileroots[0].split('_')[0] + '_' + 'deepframe.fits'
            deepframe = fits.getdata(outdir + existing_deep, 0)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = badpixstep(results, output_dir=outdir,
                                 save_results=save_results, max_iter=max_iter)
                results, deepframe = res[0], res[2]
    else:
        deepframe = None

    return results, tracemask, deepframe


if __name__ == "__main__":
    indir = 'pipeline_outputs_directory/Stage1/FirstPass/'
    input_files = utils.unpack_input_directory(indir, filetag='gainscalestep',
                                               process_f277w=False)
    background_file = 'model_background256.npy'
    background_model = np.load(background_file)

    clear_segments, f277w_segments = input_files[0], input_files[1]
    all_exposures = {'CLEAR': clear_segments}
    print('\nIdentified {} CLEAR exposure segment(s):'.format(
        len(clear_segments)))
    for file in clear_segments:
        print(' ' + file)
    if len(f277w_segments) != 0:
        all_exposures['F277W'] = f277w_segments
        print('and {} F277W exposre segment(s):'.format(len(f277w_segments)))
        for file in f277w_segments:
            print(' ' + file)

    result = run_stage2(input_files, iteration=1,
                        background_model=background_model,
                        save_results=True, force_redo=False, show_plots=False)
    stage2_results, trace_mask, deepframe = result
