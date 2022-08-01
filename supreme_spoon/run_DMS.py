#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 11:12 2022

@author: MCR

Script to run JWST DMS with custom reduction steps.
"""

from astropy.io import fits
import glob
import numpy as np

from supreme_spoon import custom_stage1, custom_stage2, custom_stage3
from supreme_spoon import utils

# ================== User Input ========================
# Stage 1 Input Files
root_dir = './'  # Root file directory
input_dir = root_dir + 'DMS_uncal/'  # Input data file directory.
input_filetag = 'uncal'  # Input file tag.
outlier_maps = None  # For 1/f correction; outlier pixel maps.
trace_mask = None  # For 1/f correcton; trace mask.
trace_mask2 = None  # For 1/f correcton; trace mask for window subtraction.

# Stage 2 Input Files
# Using STScI background model from here:
# https://jwst-docs.stsci.edu/jwst-calibration-pipeline-caveats/jwst-time-series-observations-pipeline-caveats/niriss-time-series-observation-pipeline-caveats#NIRISSTimeSeriesObservationPipelineCaveats-SOSSskybackground
background_file = root_dir + 'model_background256.npy'  # Background model.

# Stage 3 Input Files
specprofile = None  # Specprofile reference file for ATOCA.
soss_estimate = None  # SOSS estmate file for ATOCA.

# Other Parameters
output_tag = ''  # Name tag for output file directory.
run_stages = [1, 2, 3]  # Pipeline stages to run.
save_results = True  # Save results of each intermediate step to file.
show_plots = False  # Show plots.
exposure_type = 'CLEAR'  # Either CLEAR or F277W.
force_redo = False  # Force redo of steps which have already been completed.
extract_method = 'box'  # Extraction method, box or atoca.
out_frames = [50, -50]  # Out of transit frames.
# ======================================================

import os
os.environ['CRDS_PATH'] = root_dir + 'crds_cache'
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

# Unpack all files in the input directory.
input_files = utils.unpack_input_directory(input_dir, filetag=input_filetag,
                                           exposure_type=exposure_type)
print('\nIdentified {0} {1} exposure segments'.format(len(input_files), exposure_type))
for file in input_files:
    print(' ' + file)

# Run Stage 1
if 1 in run_stages:
    stage1_results = custom_stage1.run_stage1(input_files,
                                              save_results=save_results,
                                              outlier_maps=outlier_maps,
                                              trace_mask=trace_mask,
                                              trace_mask2=trace_mask2,
                                              force_redo=force_redo,
                                              root_dir=root_dir,
                                              output_tag=output_tag)
else:
    stage1_results = input_files

# Run Stage 2
if 2 in run_stages:
    background_model = np.load(background_file)
    stage2_results = custom_stage2.run_stage2(stage1_results,
                                              background_model=background_model,
                                              save_results=save_results,
                                              force_redo=force_redo,
                                              show_plots=show_plots,
                                              root_dir=root_dir,
                                              output_tag=output_tag)
    stage2_results, deepframe = stage2_results
else:
    stage2_results = input_files
    deep_file = glob.glob(root_dir + 'pipeline_outputs_directory/Stage2/*deepframe*')
    deepframe = fits.getdata(deep_file)


# Run Stage 3
if 3 in run_stages:
    stage3_results = custom_stage3.run_stage3(stage2_results,
                                              deepframe=deepframe,
                                              save_results=save_results,
                                              show_plots=show_plots,
                                              root_dir=root_dir,
                                              force_redo=force_redo,
                                              extract_method=extract_method,
                                              specprofile=specprofile,
                                              out_frames=out_frames,
                                              soss_estimate=soss_estimate,
                                              output_tag=output_tag)
    stellar_spectra = stage3_results

print('Done')
