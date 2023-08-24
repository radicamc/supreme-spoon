#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 11:12 2022

@author: MCR

Script to run JWST DMS with custom reduction steps.
"""

from datetime import datetime
import numpy as np
import os
import pandas as pd
import shutil
import sys

from supreme_spoon.stage1 import run_stage1
from supreme_spoon.stage2 import run_stage2
from supreme_spoon.stage3 import run_stage3
from supreme_spoon.utils import fancyprint, parse_config, unpack_input_dir, \
    verify_path

# Read config file.
try:
    config_file = sys.argv[1]
except IndexError:
    raise FileNotFoundError('Config file must be provided')
config = parse_config(config_file)

# Set CRDS cache path.
os.environ['CRDS_PATH'] = config['crds_cache_path']
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

# Save a copy of the config file.
if config['output_tag'] != '':
    output_tag = '_' + config['output_tag']
else:
    output_tag = config['output_tag']
root_dir = 'pipeline_outputs_directory' + output_tag
verify_path(root_dir)
root_dir += '/config_files'
verify_path(root_dir)
i = 0
copy_config = root_dir + '/' + config_file
while os.path.exists(copy_config):
    i += 1
    copy_config = root_dir + '/' + config_file
    root = copy_config.split('.yaml')[0]
    copy_config = root + '_{}.yaml'.format(i)
shutil.copy(config_file, copy_config)
# Append time at which it was run.
f = open(copy_config, 'a')
time = datetime.utcnow().isoformat(sep=' ', timespec='minutes')
f.write('\nRun at {}.'.format(time))
f.close()

# Unpack all files in the input directory.
input_files = unpack_input_dir(config['input_dir'],
                               filetag=config['input_filetag'],
                               exposure_type=config['exposure_type'])
fancyprint('\nIdentified {0} {1} exposure '
           'segments'.format(len(input_files), config['exposure_type']))
for file in input_files:
    fancyprint(' ' + file)

# Open some of the input files.
background_model = np.load(config['background_file'])
if config['smoothed_wlc'] is not None:
    config['smoothed_wlc'] = np.load(config['smoothed_wlc'])
if config['centroids'] is not None:
    config['centroids'] = pd.read_csv(config['centroids'], comment='#')
if config['f277w'] is not None:
    config['f277w'] = np.load(config['f277w'])

# ===== Run Stage 1 =====
if 1 in config['run_stages']:
    stage1_results = run_stage1(input_files, background_model=background_model,
                                baseline_ints=config['baseline_ints'],
                                smoothed_wlc=config['smoothed_wlc'],
                                save_results=config['save_results'],
                                pixel_masks=config['outlier_maps'],
                                force_redo=config['force_redo'],
                                rejection_threshold=config['rejection_threshold'],
                                flag_in_time=config['flag_in_time'],
                                time_rejection_threshold=config['time_rejection_threshold'],
                                output_tag=config['output_tag'],
                                skip_steps=config['stage1_skip'],
                                do_plot=config['do_plots'],
                                **config['stage1_kwargs'])
else:
    stage1_results = input_files

# ===== Run Stage 2 =====
if 2 in config['run_stages']:
    stage2_results = run_stage2(stage1_results,
                                background_model=background_model,
                                baseline_ints=config['baseline_ints'],
                                smoothed_wlc=config['smoothed_wlc'],
                                save_results=config['save_results'],
                                force_redo=config['force_redo'],
                                calculate_stability_ccf=config['calculate_stability_ccf'],
                                stability_params_ccf=config['stability_params_ccf'],
                                nthreads=config['nthreads'],
                                calculate_stability_pca=config['calculate_stability_pca'],
                                pca_components=config['pca_components'],
                                output_tag=config['output_tag'],
                                smoothing_scale=config['smoothing_scale'],
                                skip_steps=config['stage2_skip'],
                                generate_lc=config['generate_lc'],
                                generate_tracemask=config['generate_tracemask'],
                                mask_width=config['mask_width'],
                                pixel_flags=config['outlier_maps'],
                                generate_order0_mask=config['generate_order0_mask'],
                                f277w=config['f277w'],
                                do_plot=config['do_plots'],
                                **config['stage2_kwargs'])
else:
    stage2_results = input_files

# ===== Run Stage 3 =====
if 3 in config['run_stages']:
    stage3_results = run_stage3(stage2_results,
                                save_results=config['save_results'],
                                force_redo=config['force_redo'],
                                extract_method=config['extract_method'],
                                soss_width=config['soss_width'],
                                specprofile=config['specprofile'],
                                centroids=config['centroids'],
                                st_teff=config['st_teff'],
                                st_logg=config['st_logg'],
                                st_met=config['st_met'],
                                planet_letter=config['planet_letter'],
                                output_tag=config['output_tag'],
                                do_plot=config['do_plots'])

fancyprint('Done')
