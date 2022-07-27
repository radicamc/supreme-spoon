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
        if isinstance(file, str):
            filename = file
        else:
            filename = file.meta.filename
        filename_split = filename.split('/')[-1].split('_')
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
            newdata[itg] = utils.do_replacement(newdata[itg], to_replace,
                                                box_size=box_size)
        it += 1

    # Ensure that the bad pixels mask remains zeros or ones.
    badpix_mask = np.where(badpix_mask == 0, 0, 1)
    # Generate a final corrected deep frame.
    deepframe = utils.make_deepstack(newdata)

    current_int = 0
    # Save interpolated data.
    for n, file in enumerate(data):
        currentdata = file.data
        nints = np.shape(currentdata)[0]
        file.data = newdata[current_int:(current_int + nints)]
        current_int += nints
        if save_results is True:
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


def backgroundstep(datafiles, background_model, output_dir=None,
                   save_results=True, show_plots=False):

    print('Starting custom background subtraction step.')
    # Output directory formatting.
    if output_dir is not None:
        if output_dir[-1] != '/':
            output_dir += '/'

    datafiles = np.atleast_1d(datafiles)
    opened_datafiles = []
    for i, file in enumerate(datafiles):
        if isinstance(file, str):
            currentfile = datamodels.open(file)
        else:
            currentfile = file
        opened_datafiles.append(currentfile)
        # To create the deepstack, join all segments together.
        if i == 0:
            cube = currentfile.data
        else:
            cube = np.concatenate([cube, currentfile.data], axis=0)
    datafiles = opened_datafiles
    deepstack = utils.make_deepstack(cube)

    # Do model scaling
    scale_mod = np.nanmedian(background_model[210:250, 500:800])
    scale_dat = np.nanmedian(deepstack[210:250, 500:800])
    model_scaled = background_model / scale_mod * scale_dat

    results = []
    for currentfile in datafiles:
        old_filename = currentfile.meta.filename
        to_remove = old_filename.split('_')[-1]
        fileroot = old_filename.split(to_remove)[0]

        data_backsub = currentfile.data - model_scaled

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
        results.append(currentfile)
        currentfile.close()

    return results


def run_stage2(results, background_model=None, save_results=True,
               force_redo=False, show_plots=False, max_iter=2, mask_width=30,
               root_dir='./'):
    # ============== DMS Stage 2 ==============
    # Spectroscopic processing.
    # Documentation: https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_spec2.html
    print('\n\n**Starting supreme-SPOON Stage 2**')
    print('Spectroscopic processing\n\n')

    utils.verify_path(root_dir + 'pipeline_outputs_directory')
    utils.verify_path(root_dir + 'pipeline_outputs_directory/Stage2')
    outdir = root_dir + 'pipeline_outputs_directory/Stage2/'

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

    # ===== Bad Pixel Correction Step =====
    # Custom DMS step.
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
        # Get total file root, with no segment info.
        working_name = fileroots[0]
        if 'seg' in working_name:
            parts = working_name.split('seg')
            part1, part2 = parts[0][:-1], parts[1][3:]
            fileroot_noseg = part1 + part2
        else:
            fileroot_noseg = fileroots[0]
        existing_deep = fileroot_noseg + 'deepframe.fits'
        deepframe = fits.getdata(outdir + existing_deep, 0)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = badpixstep(results, output_dir=outdir,
                             save_results=save_results, max_iter=max_iter)
            results, deepframe = res[0], res[2]

    return results, deepframe


if __name__ == "__main__":
    # =============== User Input ===============
    root_dir = '/home/radica/jwst/ERO/WASP-96b/'
    indir = root_dir + 'pipeline_outputs_directory/Stage1/'
    input_filetag = 'gainscalestep'
    background_file = root_dir + 'model_background256.npy'
    # ==========================================

    import os
    os.environ['CRDS_PATH'] = root_dir + 'crds_cache'
    os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

    input_files = utils.unpack_input_directory(indir, filetag=input_filetag,
                                               process_f277w=False)
    background_model = np.load(background_file)

    clear_segments, f277w_segments = input_files[0], input_files[1]
    all_exposures = {'CLEAR': clear_segments}
    print('\nIdentified {} CLEAR exposure segment(s):'.format(len(clear_segments)))
    for file in clear_segments:
        print(' ' + file)
    if len(f277w_segments) != 0:
        all_exposures['F277W'] = f277w_segments
        print('and {} F277W exposre segment(s):'.format(len(f277w_segments)))
        for file in f277w_segments:
            print(' ' + file)

    result = run_stage2(all_exposures['CLEAR'],
                        background_model=background_model,
                        save_results=True, force_redo=False, show_plots=False,
                        root_dir=root_dir)
    stage2_results, deepframe = result
