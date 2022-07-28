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
from jwst.extract_1d.soss_extract import soss_boxextract

from supreme_spoon import utils
from supreme_spoon import plotting


def oneoverfstep(datafiles, output_dir=None, save_results=True,
                 outlier_maps=None, trace_mask=None, trace_mask2=None,
                 use_dq=True):
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
    outlier_maps : list[str], None
        List of paths to outlier maps for each data segment. Can be
        3D (nints, dimy, dimx), or 2D (dimy, dimx) files.
    trace_mask : str, None
        Path to trace mask file. Should be 2D (dimy, dimx).
    trace_mask2 : str, None
        Path to trace mask file. Should be 3D (norder, dimy, dimx). If provided 1/f
        will also be subtracted between trace_mask and trace_mask2.
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

    # Generate the deep stack and rms of it. Both 3D (ngroup, dimy, dimx).
    print('Generating a deep stack for each frame using all integrations...')
    deepstack = utils.make_deepstack(cube)
    # Save these to disk if requested.

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
                outliers = np.repeat(outliers, nint).reshape(
                    (dimy, dimx, nint))
                outliers = outliers.transpose(2, 0, 1)

        # Read in the trace mask -- a (dimy, dimx) data frame.
        if trace_mask is None:
            msg = ' No trace mask passed, ignoring the trace.'
            print(msg)
            tracemask1 = np.zeros((3, dimy, dimx))
        else:
            if isinstance(trace_mask, str):
                print(' Using trace mask {}'.format(trace_mask))
                tracemask1 = fits.getdata(trace_mask)
            else:
                print(' Using a trace mask')
                tracemask1 = trace_mask

        if trace_mask2 is not None:
            print(' Also subtracting a median around the trace')
            if isinstance(trace_mask2, str):
                tracemask2 = fits.getdata(trace_mask2)
            else:
                tracemask2 = trace_mask2
            window = True
        else:
            window = False

        # The outlier map is 0 where good and >0 otherwise. As this will be
        # applied multiplicatively, replace 0s with 1s and others with NaNs.
        outliers = np.where(outliers == 0, 1, np.nan)
        # Same thing with the trace mask.
        tracemask = tracemask1[0].astype(bool) | tracemask1[1].astype(bool) | tracemask1[2].astype(bool)
        tracemask = np.where(tracemask == 0, 1, np.nan)
        tracemask = np.repeat(tracemask, nint).reshape((dimy, dimx, nint))
        tracemask = tracemask.transpose(2, 0, 1)
        # Combine the two masks.
        outliers = (outliers + tracemask) // 2

        dcmap = np.copy(datamodel.data)
        dcmap_w = np.zeros((2, nint, ngroup, dimy, dimx))
        subcorr = np.copy(datamodel.data)
        subcorr_w = np.copy(datamodel.data)
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
                if datamodel.meta.subarray.name == 'SUBSTRIP256':
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore',
                                              category=RuntimeWarning)
                        dc = np.nanmedian(sub_m[i, g], axis=0)
                    # dc is 1D (columns) - expand to 2D
                    dc2d = np.repeat(dc, 256).reshape((2048, 256)).transpose(1, 0)
                    dcmap[i, g, :, :] = dc2d
                    subcorr[i, g, :, :] = sub[i, g, :, :] - dcmap[i, g, :, :]

                    if window is True:
                        for order in [0, 1]:
                            trace_window = tracemask2[order] - tracemask1[order]
                            trace_window = np.where(trace_window == 0, 1, np.nan)
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore',
                                                      category=RuntimeWarning)
                                dc_w = np.nanmedian(subcorr[i, g, :, :] * trace_window, axis=0)
                            dc2d_w = np.repeat(dc_w, 256).reshape((2048, 256)).transpose(1, 0)
                            dcmap_w[order, i, g, :, :] = dc2d_w * tracemask2[order]
                            subcorr_w[i, g, :, :] = subcorr[i, g, :, :] - dcmap_w[order, i, g, :, :]

                else:
                    raise NotImplementedError

        # Make sure no NaNs are in the DC map
        dcmap = np.where(np.isfinite(dcmap), dcmap, 0)

        # Subtract the DC map from a copy of the data model
        rampmodel_corr = datamodel.copy()
        corr_data = datamodel.data - dcmap
        if window is True:
            corr_data = corr_data - dcmap_w[0] - dcmap_w[1]
            subcorr = np.stack([subcorr, subcorr_w])
            dcmap = np.stack([dcmap, dcmap_w[0], dcmap_w[1]])
        rampmodel_corr.data = corr_data

        # Save results to disk if requested.
        if save_results is True:
            hdu = fits.PrimaryHDU(sub)
            hdu.writeto(output_dir + fileroots[n] + 'oneoverfstep_diffim.fits',
                        overwrite=True)
            hdu = fits.PrimaryHDU(subcorr)
            hdu.writeto(
                output_dir + fileroots[n] + 'oneoverfstep_diffimcorr.fits',
                overwrite=True)
            hdu = fits.PrimaryHDU(dcmap)
            hdu.writeto(
                output_dir + fileroots[n] + 'oneoverfstep_noisemap.fits',
                overwrite=True)
            corrected_rampmodels.append(rampmodel_corr)
            rampmodel_corr.write(
                output_dir + fileroots[n] + 'oneoverfstep.fits')

        datamodel.close()

    return corrected_rampmodels


def tracemaskstep(deepframe, output_dir, mask_width=30, save_results=True,
                  show_plots=False):

    if isinstance(deepframe, str):
        deepframe = datamodels.open(deepframe)

    fileroot = deepframe.meta.filename.split('deepframe')[0]
    deepframe = deepframe.extra_fits.PRIMARY.data

    # Get orders 1 to 3 centroids
    dimy, dimx = np.shape(deepframe)
    if dimy == 256:
        subarray = 'SUBSTRIP256'
    else:
        raise NotImplementedError

    cen_o1, cen_o2, cen_o3 = utils.get_trace_centroids(deepframe, subarray,
                                                       output_dir=output_dir,
                                                       save_results=save_results,
                                                       save_filename=fileroot)
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

    tracemask = np.zeros((3, dimy, dimx))
    tracemask[0] = weights1
    tracemask[1] = weights2
    tracemask[2] = weights3
    if show_plots is True:
        plotting.do_tracemask_plot(weights1 | weights2 | weights3)

    if save_results is True:
        hdu = fits.PrimaryHDU(tracemask)
        hdu.writeto(output_dir + fileroot + 'tracemask_width{}.fits'.format(mask_width),
                    overwrite=True)

    return tracemask


def run_stage1(results, save_results=True, outlier_maps=None, trace_mask=None,
               trace_mask2=None, force_redo=False, rejection_threshold=5,
               root_dir='./'):
    # ============== DMS Stage 1 ==============
    # Detector level processing.
    # Documentation: https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_detector1.html
    print('\n\n**Starting supreme-SPOON Stage 1**')
    print('Detector level processing\n\n')

    utils.verify_path(root_dir + 'pipeline_outputs_directory')
    utils.verify_path(root_dir + 'pipeline_outputs_directory/Stage1')
    outdir = root_dir + 'pipeline_outputs_directory/Stage1/'

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

    # ===== Group Scale Step =====
    # Default DMS step.
    step_tag = 'groupscalestep.fits'
    new_results = []
    for i, segment in enumerate(results):
        expected_file = outdir + fileroots[i] + step_tag
        if expected_file in all_files and force_redo is False:
            print('Output file {} already exists.'.format(expected_file))
            print('Skipping Group Scale Step.\n')
            res = expected_file
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
        expected_file = outdir + fileroots[i] + step_tag
        if expected_file in all_files and force_redo is False:
            print('Output file {} already exists.'.format(expected_file))
            print('Skipping Data Quality Initialization Step.\n')
            res = expected_file
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
        expected_file = outdir + fileroots[i] + step_tag
        if expected_file in all_files and force_redo is False:
            print('Output file {} already exists.'.format(expected_file))
            print('Skipping Saturation Detection Step.\n')
            res = expected_file
        else:
            step = calwebb_detector1.saturation_step.SaturationStep()
            res = step.call(segment, output_dir=outdir,
                            save_results=save_results)
        new_results.append(res)
    results = new_results

    # ===== 1/f Noise Correction Step =====
    # Custom DMS step.
    # On the second iteration, include bad pixel and trace masks to
    # improve the 1/f noise estimation.
    step_tag = 'oneoverfstep.fits'
    do_step = 1
    new_results = []
    for i in range(len(results)):
        expected_file = outdir + fileroots[i] + step_tag
        if expected_file not in all_files:
            do_step *= 0
        else:
            new_results.append(outdir + expected_file)
    if do_step == 1 and force_redo is False:
        print('Output files already exist.')
        print('Skipping 1/f Correction Step.\n')
        results = new_results
    else:
        results = oneoverfstep(results, output_dir=outdir,
                               save_results=save_results,
                               outlier_maps=outlier_maps,
                               trace_mask=trace_mask, trace_mask2=trace_mask2)

    # ===== Superbias Subtraction Step =====
    # Default DMS step.
    step_tag = 'superbiasstep.fits'
    new_results = []
    for i, segment in enumerate(results):
        expected_file = outdir + fileroots[i] + step_tag
        if expected_file in all_files and force_redo is False:
            print('Output file {} already exists.'.format(expected_file))
            print('Skipping Superbias Subtraction Step.\n')
            res = expected_file
        else:
            step = calwebb_detector1.superbias_step.SuperBiasStep()
            res = step.call(segment, output_dir=outdir,
                            save_results=save_results)
            # Hack to fix file name
            res = utils.fix_filenames(res, '_oneoverfstep_', outdir)[0]
        new_results.append(res)
    results = new_results

    # ===== Linearity Correction Step =====
    # Default DMS step.
    step_tag = 'linearitystep.fits'
    new_results = []
    for i, segment in enumerate(results):
        expected_file = outdir + fileroots[i] + step_tag
        if expected_file in all_files and force_redo is False:
            print('Output file {} already exists.'.format(expected_file))
            print('Skipping Linearity Correction Step.\n')
            res = expected_file
        else:
            step = calwebb_detector1.linearity_step.LinearityStep()
            res = step.call(segment, output_dir=outdir,
                            save_results=save_results)
        new_results.append(res)
    results = new_results

    # ===== Jump Detection Step =====
    # Default DMS step.
    step_tag = 'jump.fits'
    new_results = []
    for i, segment in enumerate(results):
        expected_file = outdir + fileroots[i] + step_tag
        if expected_file in all_files and force_redo is False:
            print('Output file {} already exists.'.format(expected_file))
            print('Skipping Jump Detection Step.\n')
            res = expected_file
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
        expected_file = outdir + fileroots[i] + step_tag
        if expected_file in all_files and force_redo is False:
            print('Output file {} already exists.'.format(expected_file))
            print('Skipping Ramp Fit Step.\n')
            res = expected_file
        else:
            step = calwebb_detector1.ramp_fit_step.RampFitStep()
            res = step.call(segment, output_dir=outdir,
                            save_results=save_results)[1]
            # Store pixel flags in seperate files to be used for 1/f noise
            # correction.
            hdu = fits.PrimaryHDU(res.dq)
            outfile = outdir + fileroots[i] + 'dqpixelflags.fits'
            hdu.writeto(outfile, overwrite=True)
            # Hack to fix file names
            res = utils.fix_filenames(res, '_1_', outdir)[0]
        new_results.append(res)
    results = new_results

    # ===== Gain Scale Correcton Step =====
    # Default DMS step.
    step_tag = 'gainscalestep.fits'
    new_results = []
    for i, segment in enumerate(results):
        expected_file = outdir + fileroots[i] + step_tag
        if expected_file in all_files and force_redo is False:
            print('Output file {} already exists.'.format(expected_file))
            print('Skipping Gain Scale Correction Step.\n')
            res = expected_file
        else:
            step = calwebb_detector1.gain_scale_step.GainScaleStep()
            res = step.call(segment, output_dir=outdir,
                            save_results=save_results)
        new_results.append(res)
    results = new_results

    return results


if __name__ == "__main__":
    # =============== User Input ===============
    root_dir = '/home/radica/jwst/ERO/WASP-96b/'
    indir = root_dir + 'DMS_uncal/'
    input_filetag = 'uncal'
    outlier_maps = ['OLD_pipeline_outputs_directory/Stage1/jw02734002001_04101_00001-seg001_nis_1_dqpixelflags.fits',
                   'OLD_pipeline_outputs_directory/Stage1/jw02734002001_04101_00001-seg002_nis_1_dqpixelflags.fits',
                   'OLD_pipeline_outputs_directory/Stage1/jw02734002001_04101_00001-seg003_nis_1_dqpixelflags.fits']
    trace_mask = 'OLD_pipeline_outputs_directory/Stage2/jw02734002001_tracemask.fits'
    trace_mask2 = None
    # ==========================================

    import os
    os.environ['CRDS_PATH'] = root_dir + 'crds_cache'
    os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

    input_files = utils.unpack_input_directory(indir, filetag=input_filetag,
                                               process_f277w=False)

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

    stage1_results = run_stage1(all_exposures['CLEAR'], save_results=True,
                                outlier_maps=outlier_maps,
                                trace_mask=trace_mask, trace_mask2=trace_mask2,
                                force_redo=False, root_dir=root_dir)
