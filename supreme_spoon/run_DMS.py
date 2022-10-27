#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 11:12 2022

@author: MCR

Script to run JWST DMS with custom reduction steps.
"""

import numpy as np
import os
import sys

from supreme_spoon import stage1, stage2, stage3
from supreme_spoon import utils

os.environ['CRDS_PATH'] = './crds_cache'
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

# Read config file.
try:
    config_file = sys.argv[1]
except IndexError:
    msg = 'Config file must be provided'
    raise FileNotFoundError(msg)
config = utils.parse_config(config_file)


# Unpack all files in the input directory.
input_files = utils.unpack_input_directory(config['input_dir'],
                                           filetag=config['input_filetag'],
                                           exposure_type=config['exposure_type'])
print('\nIdentified {0} {1} exposure segments'.format(len(input_files),
                                                      config['exposure_type']))
for file in input_files:
    print(' ' + file)

# === Run Stage 1 ===
if 1 in config['run_stages']:
    background_model = np.load(config['background_file'])
    if config['smoothed_wlc'] is not None:
        config['smoothed_wlc'] = np.load(config['smoothed_wlc'])
    stage1_results = stage1.run_stage1(input_files,
                                       background_model=background_model,
                                       baseline_ints=config['baseline_ints'],
                                       smoothed_wlc=config['smoothed_wlc'],
                                       save_results=config['save_results'],
                                       outlier_maps=config['outlier_maps'],
                                       trace_mask=config['trace_mask'],
                                       even_odd_rows=config['even_odd_rows'],
                                       force_redo=config['force_redo'],
                                       output_tag=config['output_tag'],
                                       occultation_type=config['occultation_type'])
else:
    stage1_results = input_files

# === Run Stage 2 ===
if 2 in config['run_stages']:
    background_model = np.load(config['background_file'])
    results = stage2.run_stage2(stage1_results,
                                smoothed_wlc=config['smoothed_wlc'],
                                background_model=background_model,
                                baseline_ints=config['baseline_ints'],
                                save_results=config['save_results'],
                                force_redo=config['force_redo'],
                                output_tag=config['output_tag'],
                                occultation_type=config['occultation_type'],
                                mask_width=config['mask_width'],
                                calculate_stability=config['calculate_stability'],
                                stability_params=config['stability_params'],
                                smoothing_scale=config['smoothing_scale'])
    stage2_results = results[0]
    deepframe = results[1]
    centroids = results[3]
    smoothed_wlc = results[4]
else:
    # Get the existing secondary outputs from Stage 2 necessary for Stage 3.
    stage2_results = input_files
    stage2_secondary = utils.open_stage2_secondary_outputs(config['deepframe'],
                                                           config['centroids'],
                                                           config['smoothed_wlc'],
                                                           config['output_tag'])
    deepframe, centroids, smoothed_wlc = stage2_secondary

# === Run Stage 3 ===
if 3 in config['run_stages']:
    stage3_results = stage3.run_stage3(stage2_results,
                                       deepframe=deepframe,
                                       smoothed_wlc=smoothed_wlc,
                                       baseline_ints=config['baseline_ints'],
                                       save_results=config['save_results'],
                                       force_redo=config['force_redo'],
                                       extract_method=config['extract_method'],
                                       soss_width=config['soss_width'],
                                       specprofile=config['specprofile'],
                                       soss_estimate=config['soss_estimate'],
                                       output_tag=config['output_tag'],
                                       use_applesoss=config['use_applesoss'],
                                       occultation_type=config['occultation_type'],
                                       soss_tikfac=config['soss_tikfac'])
    stellar_spectra = stage3_results

print('Done')
