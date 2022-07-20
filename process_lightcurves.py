#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 13:59 2022

@author: MCR

Lightcurve processing routines.
"""

import numpy as np


def sigma_clip_lightcurves(flux, ferr, thresh=5):
    flux_clipped = np.copy(flux)
    nints, nwave = np.shape(flux)
    clipsum = 0
    for itg in range(nints):
        med = np.nanmedian(flux[itg])
        ii = np.where(np.abs(flux[itg] - med) / ferr[itg] > thresh)[0]
        flux_clipped[itg, ii] = med
        clipsum += len(ii)

    print('{0} pixels clipped ({1:.3f}%)'.format(clipsum, clipsum / nints / nwave * 100))

    return flux_clipped
