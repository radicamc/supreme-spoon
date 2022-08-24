#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 11:12 2022

@author: MCR

Script to run JWST DMS with custom reduction steps.
"""

import numpy as np

from supreme_spoon import stage1, stage2, custom_stage3
from supreme_spoon import utils

# ================== User Input ========================
# === Key Parameters ===
# Root file directory
root_dir = './'
# Input data file directory.
input_dir = root_dir + 'DMS_uncal/'
# Input file tag.
input_filetag = 'uncal'

# === Stage 1 Input Files & Parameters ===
# For 1/f correction; outlier pixel maps (optional).
outlier_maps = None
# For 1/f correcton; trace mask (optional).
trace_mask = None
# For 1/f correction; estimate of white light curve (optional).
scaling_curve = None
# Background model. Using STScI background model from here:
# https://jwst-docs.stsci.edu/jwst-calibration-pipeline-caveats/jwst-time-series-observations-pipeline-caveats/niriss-time-series-observation-pipeline-caveats#NIRISSTimeSeriesObservationPipelineCaveats-SOSSskybackground
background_file = root_dir + 'model_background256.npy'

# === Stage 2 Input Parameters ===
# Timescale on which to smooth lightcurve estimate  (optional).
smoothing_scale = None
# Size of box to mask for 1/f correction. Should be wider than extraction box.
mask_width = 30

# === Stage 3 Input Files & Parameters ===
# Specprofile reference file for ATOCA (optional).
specprofile = None
# SOSS estmate file for ATOCA (optional).
soss_estimate = None
# Median stack of the TSO (optional; not necessary if running Stage 2).
deepframe = None
# Centroids for all three orders (optional; not necessary if running Stage 2).
centroids = None

# === Other Parameters ===
# Name tag for output file directory.
output_tag = ''
# Pipeline stages to run.
run_stages = [1, 2, 3]
# Type of exposure; either CLEAR or F277W.
exposure_type = 'CLEAR'
# Extraction method, box or atoca.
extract_method = 'box'
# Save results of each intermediate step to file.
save_results = True
# Force redo of steps which have already been completed.
force_redo = False
# Integrations of ingress and egress.
baseline_ints = [50, -50]
# Type of occultation: 'transit' or 'eclipse'.
occultation_type = 'transit'
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

# === Run Stage 1 ===
if 1 in run_stages:
    background_model = np.load(background_file)
    if scaling_curve is not None:
        scaling_curve = np.load(scaling_curve)
    stage1_results = stage1.run_stage1(input_files,
                                       background_model=background_model,
                                       baseline_ints=baseline_ints,
                                       smoothed_wlc=scaling_curve,
                                       save_results=save_results,
                                       outlier_maps=outlier_maps,
                                       trace_mask=trace_mask,
                                       force_redo=force_redo,
                                       root_dir=root_dir,
                                       output_tag=output_tag,
                                       occultation_type=occultation_type)
else:
    stage1_results = input_files

# === Run Stage 2 ===
if 2 in run_stages:
    background_model = np.load(background_file)
    results = stage2.run_stage2(stage1_results,
                                baseline_ints=baseline_ints,
                                save_results=save_results,
                                force_redo=force_redo, root_dir=root_dir,
                                output_tag=output_tag,
                                occultation_type=occultation_type,
                                mask_width=mask_width,
                                smoothing_scale=smoothing_scale)
    stage2_results, deepframe, centroids = results[0], results[1], results[3]
elif 3 in run_stages:
    # Get the existing secondary outputs from Stage 2 necessary for Stage 3.
    stage2_results = input_files
    deepframe, centroids = utils.open_stage2_secondary_outputs(deepframe,
                                                               centroids,
                                                               root_dir,
                                                               output_tag)

# === Run Stage 3 ===
if 3 in run_stages:
    stage3_results = custom_stage3.run_stage3(stage2_results,
                                              deepframe=deepframe,
                                              save_results=save_results,
                                              root_dir=root_dir,
                                              force_redo=force_redo,
                                              extract_method=extract_method,
                                              specprofile=specprofile,
                                              out_frames=baseline_ints,
                                              soss_estimate=soss_estimate,
                                              output_tag=output_tag)
    stellar_spectra = stage3_results
else:
    pass

print('Done')
