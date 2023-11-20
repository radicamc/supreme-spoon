#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 15:08 2023

@author: MCR

Extra functions to be used alongside the main pipeline.
"""

from astropy.io import fits
import numpy as np


def get_throughput_from_photom_file(photom_path):
    """Calculate the throughput based on the photom reference file.
    Function from Loïc, and is apparently the proper way to get the
    throughput? Gives different results from contents of spectra reference
    file.

    Parameters
    ----------
    photom_path : str
        Path to photom reference file

    Returns
    -------
    w1 : np.array(float)
        Order 1 wavelength axis.
    w2 : np.array(float)
        Order 2 wavelength axis.
    thpt1 : np.array(float)
        Order 1 throughput values
    thpt2 : np.array(float)
        Order 2 throughput values
    """

    # From the photom ref file, get conversion factors + wavelength/pixel grid.
    photom = fits.open(photom_path)
    w1 = photom[1].data['wavelength'][0]
    ii = np.where((photom[1].data['wavelength'][0] >= 0.84649785) &
                  (photom[1].data['wavelength'][0] <= 2.83358154))
    w1 = w1[ii]
    scale1 = photom[1].data['relresponse'][0][ii]

    w2 = photom[1].data['wavelength'][1]
    ii = np.where((photom[1].data['wavelength'][1] >= 0.49996997) &
                  (photom[1].data['wavelength'][1] <= 1.40884607))
    w2 = w2[ii]
    scale2 = photom[1].data['relresponse'][1][ii]

    # Calculate throughput from conversion factor.
    thpt1 = 1 / (scale1 * 3e8 / w1 ** 2)
    thpt2 = 1 / (scale2 * 3e8 / w2 ** 2)

    return w1, w2, thpt1, thpt2
