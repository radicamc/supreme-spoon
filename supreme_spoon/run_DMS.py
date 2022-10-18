#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 11:12 2022

@author: MCR

Script to run JWST DMS with custom reduction steps.
"""
import numpy as np

from supreme_spoon import stage1, stage2, stage3
from supreme_spoon import utils

# ================== User Input ========================
# ===== Key Parameters =====
# Root file directory
root_dir = './'
# Input data file directory.
input_dir = root_dir + 'DMS_uncal/'
# Input file tag.
input_filetag = 'uncal'

# ===== Stage 1 Input Files & Parameters =====
# For 1/f correction; outlier pixel maps (optional).
outlier_maps = None
# For 1/f correcton; trace mask (optional).
trace_mask = None
# For 1/f correction; estimate of white light curve (optional).
smoothed_wlc = None
# Background model. Using STScI background model from here:
# https://jwst-docs.stsci.edu/jwst-calibration-pipeline-caveats/jwst-time-series-observations-pipeline-caveats/niriss-time-series-observation-pipeline-caveats#NIRISSTimeSeriesObservationPipelineCaveats-SOSSskybackground
background_file = root_dir + 'model_background256.npy'
# For 1/f correction; treat even and odd numbered rows seperately.
even_odd_rows = True

# ===== Stage 2 Input Parameters =====
# Timescale on which to smooth lightcurve estimate  (optional).
smoothing_scale = None
# Size of box to mask for 1/f correction. Should be wider than extraction box.
mask_width = 30
# If True, calculate the stability of the SOSS trace over the course of the
# TSO. These parameters can be useful for lightcurve detrending.
calculate_stability = True
# Parameters for which to calcuate the stability: 'x', 'y', 'FWHM, or 'ALL'.
stability_params = 'ALL'

# ===== Stage 3 Input Files & Parameters =====
# Specprofile reference file for ATOCA (optional).
specprofile = None
# SOSS estmate file for ATOCA (optional).
soss_estimate = None
# Median stack of the TSO (optional; not necessary if running Stage 2).
deepframe = None
# Centroids for all three orders (optional; not necessary if running Stage 2).
centroids = None
# Box width to extract around the trace center.
soss_width = 25
# Tikhonov regularization factor (optional).
soss_tikfac = None

# ===== Other General Parameters =====
# Name tag for output file directory.
output_tag = ''
# Pipeline stages to run.
run_stages = [1, 2, 3]
# Type of exposure; either CLEAR or F277W.
exposure_type = 'CLEAR'
# Extraction method, box or atoca.
extract_method = 'box'
# For ATOCA extractions only: if True, construct a SpecProfile reference
# tailored to this TSO. If False, use the default.
use_applesoss = True
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
    if smoothed_wlc is not None:
        smoothed_wlc = np.load(smoothed_wlc)
    stage1_results = stage1.run_stage1(input_files,
                                       background_model=background_model,
                                       baseline_ints=baseline_ints,
                                       smoothed_wlc=smoothed_wlc,
                                       save_results=save_results,
                                       outlier_maps=outlier_maps,
                                       trace_mask=trace_mask,
                                       even_odd_rows=even_odd_rows,
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
                                smoothed_wlc=smoothed_wlc,
                                background_model=background_model,
                                baseline_ints=baseline_ints,
                                save_results=save_results,
                                force_redo=force_redo, root_dir=root_dir,
                                output_tag=output_tag,
                                occultation_type=occultation_type,
                                mask_width=mask_width,
                                calculate_stability=calculate_stability,
                                stability_params=stability_params,
                                smoothing_scale=smoothing_scale)
    stage2_results = results[0]
    deepframe = results[1]
    centroids = results[3]
    smoothed_wlc = results[4]
elif 3 in run_stages:
    # Get the existing secondary outputs from Stage 2 necessary for Stage 3.
    stage2_results = input_files
    stage2_secondary = utils.open_stage2_secondary_outputs(deepframe,
                                                           centroids,
                                                           smoothed_wlc,
                                                           root_dir,
                                                           output_tag)
    deepframe, centroids, smoothed_wlc = stage2_secondary

# === Run Stage 3 ===
if 3 in run_stages:
    stage3_results = stage3.run_stage3(stage2_results, deepframe=deepframe,
                                       smoothed_wlc=smoothed_wlc,
                                       baseline_ints=baseline_ints,
                                       save_results=save_results,
                                       root_dir=root_dir,
                                       force_redo=force_redo,
                                       extract_method=extract_method,
                                       soss_width=soss_width,
                                       specprofile=specprofile,
                                       soss_estimate=soss_estimate,
                                       output_tag=output_tag,
                                       use_applesoss=use_applesoss,
                                       occultation_type=occultation_type,
                                       soss_tikfac=soss_tikfac)
    stellar_spectra = stage3_results

print('Done')
