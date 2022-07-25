#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 11:12 2022

@author: MCR

Script to run JWST DMS with custom reduction steps.
"""

import os
os.environ['CRDS_PATH'] = './crds_cache'
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

import numpy as np

from supreme_spoon import custom_stage1, custom_stage2, custom_stage3
from supreme_spoon import utils

# ================== User Input ========================
# Stage 1 Input Files
root_dir = '/home/radica/jwst/ERO/WASP-96b/'  # Root file directory
uncal_indir = root_dir + 'DMS_uncal/'  # Uncalibrated data file directory
input_filetag = 'uncal'  # Uncalibrated file tag
outlier_maps = None  # Outliers to mask in 1/f correction
trace_mask = None  # Trace pixels to mask in 1/f correction

# Stage 2 Input Files
background_file = root_dir + 'model_background256.npy'  # Background model

# Other Parameters
save_results = True  # Save results of each intermediate step to file
show_plots = False  # Show plots
process_f277w = False  # Process F277W exposures in addition to CLEAR
force_redo = False  # Force redo of steps which have already been completed
# ======================================================

# Unpack data file from input directory
input_files = utils.unpack_input_directory(uncal_indir, filetag=input_filetag,
                                           process_f277w=process_f277w)

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

# Run Stage 1
stage1_results = custom_stage1.run_stage1(all_exposures['CLEAR'],
                                          save_results=save_results,
                                          outlier_maps=outlier_maps,
                                          trace_mask=trace_mask,
                                          force_redo=force_redo,
                                          root_dir=root_dir)

# Run Stage 2
background_model = np.load(background_file)
stage2_results = custom_stage2.run_stage2(stage1_results,
                                          background_model=background_model,
                                          save_results=save_results,
                                          force_redo=force_redo,
                                          show_plots=show_plots,
                                          root_dir=root_dir)
stage2_results, deepframe = stage2_results

# Run Stage 3
stage3_results = custom_stage3.run_stage3(stage2_results,
                                          deepframe=deepframe,
                                          save_results=save_results,
                                          show_plots=show_plots,
                                          root_dir=root_dir,
                                          force_redo=force_redo)
normalized_lightcurves, stellar_spectra = stage3_results

print('Done')
