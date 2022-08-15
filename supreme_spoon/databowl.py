#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 15:34 2022

@author: MCR

Definition of supreme-SPOON DataBowl storage class.
"""

import numpy as np

from jwst import datamodels

from supreme_spoon import utils


class DataBowl:
    def __init__(self, datafiles, occultation_type='transit',
                 deepframe=None, tracemask=None, smoothed_lightcurve=None,
                 stellar_spectra=None, centroids=None, trace_profile=None,
                 norm_frames=None, background_models=None):
        datafiles = np.atleast_1d(datafiles)
        self.datamodels = []
        for file in datafiles:
            self.datamodels.append(datamodels.open(file))
        self.fileroots = self.initialize_fileroots()
        self.fileroot_noseg = self.initialize_fileroot_noseg()
        self.time = utils.get_timestamps(self.datamodels)
        self.deepframe = deepframe
        self.tracemask = tracemask
        self.smoothed_lightcurve = smoothed_lightcurve
        self.stellar_spectra = stellar_spectra
        self.centroids = centroids
        self.trace_profile = trace_profile
        self.occultation_type = occultation_type
        if norm_frames is not None:
            norm_frames = utils.format_out_frames(norm_frames,
                                                  self.occultation_type)
        self.baseline_integrations = norm_frames
        self.background_model = background_models

    def initialize_fileroots(self):
        fileroots = []
        # Load in datamodels from all segments.
        for i, file in enumerate(self.datamodels):
            # Get file name root.
            filename_split = file.meta.filename.split('/')[-1].split('_')
            fileroot = ''
            for seg, segment in enumerate(filename_split):
                if seg == len(filename_split) - 1:
                    break
                segment += '_'
                fileroot += segment
            fileroots.append(fileroot)
        return fileroots

    def initialize_fileroot_noseg(self):
        # Get total file root, with no segment info.
        working_name = self.fileroots[0]
        if 'seg' in working_name:
            parts = working_name.split('seg')
            part1, part2 = parts[0][:-1], parts[1][3:]
            fileroot_noseg = part1 + part2
        else:
            fileroot_noseg = self.fileroots[0]
        return fileroot_noseg
