#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 11:12 2022

@author: MCR

Script to run JWST DMS with custom reduction steps.
"""

# TODO: make mains for each step
import os
os.environ['CRDS_PATH'] = './crds_cache'
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

from astropy.io import fits
import numpy as np
import warnings

from jwst import datamodels
from jwst.pipeline import calwebb_detector1
from jwst.pipeline import calwebb_spec2

from supreme_spoon import custom_stage1, custom_stage2, custom_stage3
from supreme_spoon import utils

# ================== User Input ========================
uncal_indir = 'DMS_uncal/'  # Directory containing uncalibrated data files
background_file = 'model_background256.npy'  # Background model file
planet_name = 'WASP-96b'  # Name of observed planet

save_results = True  # Save results of each intermediate step to file
show_plots = False  # Show plots
process_f277w = False  # Process F277W exposures in addition to CLEAR
# ======================================================

# =================== Initial Setup ====================
print('\n\nStarting Custom JWST DMS - SUPREME-SPOON\n')

if uncal_indir[-1] != '/':
    uncal_indir += '/'

res = utils.unpack_input_directory(uncal_indir, filetag='uncal',
                                   process_f277w=process_f277w)
clear_segments, f277w_segments = res
all_exposures = {'CLEAR': clear_segments}
print('\nIdentified {} CLEAR exposure segment(s):'.format(len(clear_segments)))
for file in clear_segments:
    print(' ' + file)
if len(f277w_segments) != 0:
    all_exposures['F277W'] = f277w_segments
    print('and {} F277W exposre segment(s):'.format(len(f277w_segments)))
    for file in f277w_segments:
        print(' ' + file)

# Generate output directories
utils.verify_path('pipeline_outputs_directory')
utils.verify_path('pipeline_outputs_directory/Stage1')
utils.verify_path('pipeline_outputs_directory/Stage1/FirstPass')
utils.verify_path('pipeline_outputs_directory/Stage1/SecondPass')
utils.verify_path('pipeline_outputs_directory/Stage2')
utils.verify_path('pipeline_outputs_directory/Stage2/FirstPass')
utils.verify_path('pipeline_outputs_directory/Stage2/SecondPass')
utils.verify_path('pipeline_outputs_directory/Stage3')
# ======================================================

# ==================== Run the DMS =====================
# Process both the CLEAR and F277W exposures.
for filter in all_exposures.keys():
    print('\nProcessing {} exposure files'.format(filter))
    results = all_exposures[filter]

    # Iterate twice over the pipeline. This is to improve the 1/f correction
    # by using outlier and trace masks produced in later pipeline stages.
    for iteration in range(1, 3):
        print('\nStarting DMS Stage 1; iteration #{}.'.format(iteration))

        # ============== DMS Stage 1 ==============
        # Detector level processing.
        # Documentation: https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_detector1.html
        if iteration == 1:
            outdir = 'pipeline_outputs_directory/Stage1/FirstPass/'
        else:
            outdir = 'pipeline_outputs_directory/Stage1/SecondPass/'

        # ===== Group Scale Step =====
        # Default DMS step.
        new_results = []
        for segment in results:
            step = calwebb_detector1.group_scale_step.GroupScaleStep()
            res = step.call(segment, output_dir=outdir,
                            save_results=save_results)
            new_results.append(res)
        results = new_results

        # ===== Data Quality Initialization Step =====
        # Default DMS step.
        for segment in results:
            step = calwebb_detector1.dq_init_step.DQInitStep()
            res = step.call(segment, output_dir=outdir,
                            save_results=save_results)
            new_results.append(res)
        results = new_results

        # ===== Saturation Detection Step =====
        # Default DMS step.
        new_results = []
        for segment in results:
            step = calwebb_detector1.saturation_step.SaturationStep()
            res = step.call(segment, output_dir=outdir,
                            save_results=save_results)
            new_results.append(res)
        results = new_results

        # ===== 1/f Noise Correction Step =====
        # Custom DMS step.
        # On the second iteration, include bad pixel and trace masks to
        # improve the 1/f noise estimation.
        if iteration == 2:
            outlier_maps = dqflag_files
            trace_mask = tracemask
        else:
            outlier_maps = None
            trace_mask = None
        results = custom_stage1.oneoverfstep(results, output_dir=outdir,
                                             save_results=save_results,
                                             outlier_maps=outlier_maps,
                                             trace_mask=trace_mask)

        # ===== Superbias Subtraction Step =====
        # Default DMS step.
        new_results = []
        for segment in results:
            step = calwebb_detector1.superbias_step.SuperBiasStep()
            res = step.call(segment, output_dir=outdir,
                            save_results=save_results)
            new_results.append(res)
        results = new_results
        # Hack to fix file names
        results = utils.fix_filenames(results, 'oneoverfstep_', outdir)

        # ===== Linearity Correction Step =====
        # Default DMS step.
        new_results = []
        for segment in results:
            step = calwebb_detector1.linearity_step.LinearityStep()
            res = step.call(segment, output_dir=outdir,
                            save_results=save_results)
            new_results.append(res)
        results = new_results

        # ===== Jump Detection Step =====
        # Default DMS step.
        new_results = []
        for segment in results:
            step = calwebb_detector1.jump_step.JumpStep()
            res = step.call(segment, maximum_cores='quarter',
                            rejection_threshold=5, output_dir=outdir,
                            save_results=save_results)
            new_results.append(res)
        results = new_results

        # ===== Ramp Fit Step =====
        # Default DMS step.
        new_results = []
        for segment in results:
            step = calwebb_detector1.ramp_fit_step.RampFitStep()
            res = step.call(segment, output_dir=outdir,
                            save_results=save_results)[1]
            new_results.append(res)
        results = new_results
        # Store pixel flags in seperate files to be used for 1/f noise
        # correction.
        dqflag_files = []
        for segment in results:
            data = datamodels.open(segment)
            fileroot = data.meta.filename.split('rampfitstep')[0]
            hdu = fits.PrimaryHDU(data.dq)
            outfile = outdir + fileroot + 'dqpixelflags.fits'
            dqflag_files.append(outfile)
            hdu.writeto(outfile, overwrite=True)

        # ===== Gain Scale Correcton Step =====
        # Default DMS step.
        new_results = []
        for segment in results:
            step = calwebb_detector1.gain_scale_step.GainScaleStep()
            res = step.call(segment, output_dir=outdir,
                            save_results=save_results)
            new_results.append(res)
        results = new_results

        # ============== DMS Stage 2 ==============
        # Spectroscopic processing.
        # Documentation: https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_spec2.html
        print('\nStarting DMS Stage 2; iteration #{}.'.format(iteration))

        if iteration == 2:
            outdir = 'pipeline_outputs_directory/Stage2/FirstPass/'
        else:
            outdir = 'pipeline_outputs_directory/Stage2/SecondPass/'

        # ===== Assign WCS Step =====
        # Default DMS step.
        new_results = []
        for segment in results:
            step = calwebb_spec2.assign_wcs_step.AssignWcsStep()
            res = step.call(segment, output_dir=outdir,
                            save_results=save_results)
            new_results.append(res)
        results = new_results

        # ===== Source Type Determination Step =====
        # Default DMS step.
        new_results = []
        for segment in results:
            step = calwebb_spec2.srctype_step.SourceTypeStep()
            res = step.call(segment, output_dir=outdir,
                            save_results=save_results)
            new_results.append(res)
        results = new_results

        # ===== Flat Field Correction Step =====
        # Default DMS step.
        new_results = []
        for segment in results:
            step = calwebb_spec2.flat_field_step.FlatFieldStep()
            res = step.call(segment, output_dir=outdir,
                            save_results=save_results)
            new_results.append(res)
        results = new_results

        # ===== Background Subtraction Step =====
        # Custom DMS step.
        background_model = np.load(background_file)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            results = custom_stage2.backgroundstep(results, background_model,
                                                   output_dir=outdir,
                                                   save_results=save_results,
                                                   show_plots=show_plots)

        # ===== Construct Trace Mask =====
        # Custom DMS step.
        res = custom_stage2.tracemaskstep(results, output_dir=outdir,
                                          mask_width=30,
                                          save_results=save_results,
                                          show_plots=show_plots)
        deepframe, tracemask = res

        # ===== Bad Pixel Correction Step =====
        # Custom DMS step.
        if iteration == 2:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                results = custom_stage2.badpixstep(results, max_iter=2,
                                                   output_dir=outdir,
                                                   save_results=save_results)[0]

    # ============== DMS Stage 3 ==============
    # 1D spectral extraction.
    if filter == 'CLEAR':
        # Only extract from the CLEAR exposures.
        print('\nStarting DMS Stage 3.')
        outdir = 'pipeline_outputs_directory/Stage3/'

        # ===== 1D Extraction Step =====
        # Custom/default DMS step.
        transform = custom_stage3.get_soss_transform(deepframe, results[0],
                                                     show_plots=show_plots)
        new_results = []
        for segment in results:
            step = calwebb_spec2.extract_1d_step.Extract1dStep()
            res = step.call(segment, output_dir=outdir,
                            save_results=save_results,
                            soss_transform=[transform[0], transform[1], transform[2]],
                            soss_atoca=False, subtract_background=False,
                            soss_bad_pix='masking', soss_width=25,
                            soss_modelname=None)
            new_results.append(res)
        results = new_results
        # Hack to fix file names
        results = utils.fix_filenames(results, 'badpixstep_', outdir)

        # ===== Construct Lightcurves =====
        # Custom DMS step.
        res = custom_stage3.construct_lightcurves(results, output_dir=outdir,
                                                  save_results=save_results,
                                                  show_plots=show_plots,
                                                  planet_name=planet_name)
        normalized_lightcurves, stellar_spectra = res

print('Done')
