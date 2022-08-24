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


# TODO: reading and writing
class DataBowl:
    """Storage class for intermediate data products of the supreme-SPOON
    pipeline.

    Attributes
    ----------
    datamodels : array[jwst.datamodel]
        Datamodels for each segement of a SOSS TSO.
    time : array[float]
        Mid-integration time stamps in BJD.
    deepframe :

    stellar_spectra :

    centroids :

    trace_profile :

    background_models : array[float]
        Background models scaled to the flux level of the each group median.
    fileroots : array[str]
        File root names for each segment.
    fileroot_noseg : str
        File root name with no segment information.

    Methods
    -------

    """

    def __init__(self, datafiles, deepframe=None, stellar_spectra=None,
                 centroids=None, trace_profile=None, background_models=None):
        """Initializer for the DataBowl class.

        Parameters
        ----------
        datafiles : list[str], list[jwst.datamodel]
            Datamodoels, or paths to datamodels for all segments of a SOSS
            TSO exposure.
        deepframe :

        stellar_spectra :

        centroids :

        trace_profile :

        background_models : array[float], None
            Background models scaled to the flux level of the each group
            median.
        """

        # Load in datamnodels for each segment file.
        datafiles = np.atleast_1d(datafiles)
        self.datamodels = []
        for file in datafiles:
            self.datamodels.append(datamodels.open(file))
        # Initalize filename information.
        self.fileroots = utils.get_filename_root(self.datamodels)
        self.fileroot_noseg = utils.get_filename_root_noseg(self.fileroots)
        # Get mid-integration time stamps.
        self.time = utils.get_timestamps(self.datamodels)
        # Load in other intermediate products, if provided.
        self.deepframe = deepframe
        self.stellar_spectra = stellar_spectra
        self.centroids = centroids
        self.trace_profile = trace_profile
        self.background_models = background_models
