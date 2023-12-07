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

# ===== Setup =====
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
if np.ndim(background_model) == 3:
    background_model = background_model[0]
if config['timeseries'] is not None:
    config['timeseries'] = np.load(config['timeseries'])
if config['timeseries_o2'] is not None:
    config['timeseries_o2'] = np.load(config['timeseries_o2'])
if config['centroids'] is not None:
    config['centroids'] = pd.read_csv(config['centroids'], comment='#')
if config['f277w'] is not None:
    config['f277w'] = np.load(config['f277w'])

# Check that extraction method and 1/f correction method are compatible.
if config['extract_method'] == 'atoca':
    if config['oof_method'] not in ['scale-achromatic', 'solve']:
        raise ValueError('1/f correction method {} not compatible with '
                         'atoca extraction.'.format(config['oof_method']))

# ===== Run Stage 1 =====
if 1 in config['run_stages']:
    stage1_results = run_stage1(input_files, background_model=background_model,
                                baseline_ints=config['baseline_ints'],
                                oof_method=config['oof_method'],
                                oof_order=config['oof_order'],
                                save_results=config['save_results'],
                                pixel_masks=config['outlier_maps'],
                                force_redo=config['force_redo'],
                                rejection_threshold=config['jump_threshold'],
                                flag_in_time=config['flag_in_time'],
                                time_rejection_threshold=config['time_jump_threshold'],
                                output_tag=config['output_tag'],
                                skip_steps=config['stage1_skip'],
                                do_plot=config['do_plots'],
                                timeseries=config['timeseries'],
                                **config['stage1_kwargs'])
else:
    stage1_results = input_files

# ===== Run Stage 2 =====
if 2 in config['run_stages']:
    if config['soss_width'] <= 40:
        mask_width = 45
    elif config['soss_width'] == 'optimize':
        config['generate_tracemask'] = False
        mask_width = None
    else:
        mask_width = config['soss_width'] + 5

    stage2_results = run_stage2(stage1_results,
                                background_model=background_model,
                                baseline_ints=config['baseline_ints'],
                                save_results=config['save_results'],
                                force_redo=config['force_redo'],
                                space_thresh=config['space_outlier_threshold'],
                                time_thresh=config['time_outlier_threshold'],
                                calculate_stability=config['calculate_stability'],
                                pca_components=config['pca_components'],
                                timeseries=config['timeseries'],
                                oof_method=config['oof_method'],
                                oof_order=config['oof_order'],
                                output_tag=config['output_tag'],
                                smoothing_scale=config['smoothing_scale'],
                                skip_steps=config['stage2_skip'],
                                generate_lc=config['generate_lc'],
                                generate_tracemask=config['generate_tracemask'],
                                mask_width=mask_width,
                                pixel_masks=config['outlier_maps'],
                                generate_order0_mask=config['generate_order0_mask'],
                                f277w=config['f277w'],
                                do_plot=config['do_plots'],
                                **config['stage2_kwargs'])
else:
    stage2_results = input_files

# ===== Run Stage 3 =====
if 3 in config['run_stages']:
    if config['oof_method'] == 'window':
        window_oof = True
    else:
        window_oof = False
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
                                do_plot=config['do_plots'],
                                baseline_ints=config['baseline_ints'],
                                window_oof=window_oof,
                                oof_width=config['oof_width'],
                                timeseries_o1=config['timeseries'],
                                timeseries_o2=config['timeseries_o2'],
                                pixel_masks=config['outlier_maps'],
                                datafiles2=config['datafiles_o2'],
                                **config['stage3_kwargs'])

fancyprint('Done')
