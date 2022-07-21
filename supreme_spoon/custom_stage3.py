#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Jul 21 17:33 2022

@author: MCR

Custom JWST DMS pipeline steps for Stage 3 (1D spectral extraction).
"""

import numpy as np
import warnings

from supreme_spoon import utils
from supreme_spoon import plotting


def construct_lightcurves(datafiles, output_dir, save_results=True,
                          show_plots=False, planet_name=None):
    datafiles = np.atleast_1d(datafiles)
    dn2e = utils.get_dn2e(datafiles[0])

    for i, file in enumerate(datafiles):
        segment = utils.unpack_spectra(file)
        if i == 0:
            wave2d_o1 = segment[1]['WAVELENGTH']
            flux_o1 = segment[1]['FLUX']*dn2e
            ferr_o1 = segment[1]['FLUX_ERROR']*dn2e
            wave2d_o2 = segment[2]['WAVELENGTH']
            flux_o2 = segment[2]['FLUX']*dn2e
            ferr_o2 = segment[2]['FLUX_ERROR']*dn2e
        else:
            wave2d_o1 = np.concatenate([wave2d_o1, segment[1]['WAVELENGTH']])
            flux_o1 = np.concatenate([flux_o1, segment[1]['FLUX']*dn2e])
            ferr_o1 = np.concatenate([ferr_o1, segment[1]['FLUX_ERROR']*dn2e])
            wave2d_o2 = np.concatenate([wave2d_o2, segment[2]['WAVELENGTH']])
            flux_o2 = np.concatenate([flux_o2, segment[2]['FLUX']*dn2e])
            ferr_o2 = np.concatenate([ferr_o2, segment[2]['FLUX_ERROR']*dn2e])

    wave1d_o1, wave1d_o2 = wave2d_o1[0], wave2d_o2[0]
    t = utils.make_time_axis(file)
    out_trans = np.concatenate([np.arange(90), np.arange(40) - 40])

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        norm_factor_o1 = np.nanmedian(flux_o1[out_trans], axis=0)
        nflux_o1 = flux_o1 / norm_factor_o1
        nferr_o1 = ferr_o1 / norm_factor_o1
        norm_factor_o2 = np.nanmedian(flux_o2[out_trans], axis=0)
        nflux_o2 = flux_o2 / norm_factor_o2
        nferr_o2 = ferr_o2 / norm_factor_o2

    nflux_o1_clip = utils.sigma_clip_lightcurves(nflux_o1, nferr_o1)
    nflux_o2_clip = utils.sigma_clip_lightcurves(nflux_o2, nferr_o2)

    if show_plots is True:
        kwargs = {'vmax': 1e-4, 'vmin': -1e-4}
        plotting.plot_2dlightcurves(nflux_o1, nflux_o2, wave1d_o1, wave1d_o2)
        plotting.plot_2dlightcurves(nflux_o1 - nflux_o1_clip,
                                    nflux_o2 - nflux_o2_clip,
                                    wave1d_o1, wave1d_o2, **kwargs)
        plotting.plot_2dlightcurves(nflux_o1_clip, nflux_o2_clip, wave1d_o1,
                                    wave1d_o2)

    flux_o1_clip = nflux_o1_clip * norm_factor_o1
    flux_o2_clip = nflux_o2_clip * norm_factor_o2

    if save_results is True:
        # Save full res stellar spectra
        filename = output_dir + planet_name[:-1] + '_spectra_fullres.fits'
        header_dict, header_comments = utils.get_default_header()
        header_dict['Target_Name'] = planet_name[:-1]
        utils.write_spectra_to_file(filename, wave2d_o1, flux_o1_clip, ferr_o1,
                                    wave2d_o2, flux_o2_clip, ferr_o2, t,
                                    header_dict, header_comments)

        # Save full res lightcurves
        filename = output_dir + planet_name + '_lightcurves_fullres.fits'
        header_dict, header_comments = utils.get_default_header()
        header_dict['Target_Name'] = planet_name
        utils.write_spectra_to_file(filename, wave2d_o1, nflux_o1_clip,
                                    nferr_o1, wave2d_o2, nflux_o2_clip,
                                    nferr_o2, t, header_dict,
                                    header_comments)

    stellar_spectra = {'Wave 2D Order 1': wave2d_o1,
                       'Flux Order 1': flux_o1_clip,
                       'Flux Error Order 1': ferr_o1,
                       'Wave 2D Order 2': wave2d_o2,
                       'Flux Order 2': flux_o2_clip,
                       'Flux Error Order 2': ferr_o2,
                       'Time': t}
    normalized_lightcurves = {'Wave 2D Order 1': wave2d_o1,
                              'Flux Order 1': nflux_o1_clip,
                              'Flux Error Order 1': nferr_o1,
                              'Wave 2D Order 2': wave2d_o2,
                              'Flux Order 2': nflux_o2_clip,
                              'Flux Error Order 2': nferr_o2,
                              'Time': t}

    return normalized_lightcurves, stellar_spectra
