#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 08:51 2022

@author: MCR

Miscellaneous pipeline tools.
"""

from astropy.io import fits
from astropy.time import Time
import bottleneck as bn
import os
import numpy as np
import warnings

from jwst import datamodels
from jwst.extract_1d.soss_extract import soss_solver

from sys import path
applesoss_path = '/home/radica/GitHub/APPLESOSS/'
path.insert(1, applesoss_path)

from APPLESOSS.edgetrigger_centroids import get_soss_centroids

import plotting


def get_interp_box(data, box_size, i, j, dimx, dimy):
    """ Get median and standard deviation of a box centered on a specified
    pixel.
    """

    # Get the box limits.
    low_x = np.max([i - box_size, 0])
    up_x = np.min([i + box_size, dimx - 1])
    low_y = np.max([j - box_size, 0])
    up_y = np.min([j + box_size, dimy - 1])

    # Calculate median and std deviation of box.
    median = np.nanmedian(data[low_y:up_y, low_x:up_x])
    stddev = np.nanstd(data[low_y:up_y, low_x:up_x])

    # Pack into array.
    box_properties = np.array([median, stddev])

    return box_properties


def do_replacement(frame, badpix_map, box_size=5):
    """Replace flagged pixels with the median of a surrounding box.
    """

    dimy, dimx = np.shape(frame)
    frame_out = np.copy(frame)

    # Loop over all flagged pixels.
    for i in range(dimx):
        for j in range(dimy):
            if badpix_map[j, i] == 0:
                continue
            # If pixel is flagged, replace it with the box median.
            else:
                med = get_interp_box(frame, box_size, i, j, dimx, dimy)[0]
                frame_out[j, i] = med

    return frame_out


def make_deepstack(cube, return_rms=False):
    """Make deepstack of a TSO.
    """

    deepstack = bn.nanmedian(cube, axis=0)
    if return_rms is True:
        rms = bn.nanstd(cube, axis=0)
    else:
        rms = None

    return deepstack, rms


def unpack_spectra(filename, quantities=('WAVELENGTH', 'FLUX', 'FLUX_ERROR')):
    multi_spec = datamodels.open(filename)

    all_spec = {sp_ord: {quantity: [] for quantity in quantities}
                for sp_ord in [1, 2, 3]}
    for spec in multi_spec.spec:
        sp_ord = spec.spectral_order
        for quantity in quantities:
            all_spec[sp_ord][quantity].append(spec.spec_table[quantity])

    for sp_ord in all_spec:
        for key in all_spec[sp_ord]:
            all_spec[sp_ord][key] = np.array(all_spec[sp_ord][key])

    multi_spec.close()
    return all_spec


def make_time_axis(filepath):
    header = fits.getheader(filepath, 0)
    t_start = header['DATE-OBS'] + 'T' + header['TIME-OBS']
    tgroup = header['TGROUP'] / 3600 / 24
    ngroup = header['NGROUPS'] + 1
    nint = header['NINTS']
    t_start = Time(t_start, format='isot', scale='utc')
    t = np.arange(nint) * tgroup * ngroup + t_start.jd
    return t


def fix_filenames(old_files, to_remove, outdir):
    # Hack to fix file names
    new_files = []
    for file in old_files:
        if isinstance(file, str):
            file = datamodels.open(file)

        old_filename = file.meta.filename

        split = old_filename.split(to_remove)
        new_filename = split[0] + split[1]
        file.write(outdir + new_filename)

        new_files.append(outdir + new_filename)
        file.close()
        os.remove(outdir + old_filename)

    return new_files


def verify_path(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)


def determine_soss_transform(deepframe, spec_trace, show_plots=False):

    xref_o1 = spec_trace.trace[0].data['X']
    yref_o1 = spec_trace.trace[0].data['Y']
    xref_o2 = spec_trace.trace[1].data['X']
    yref_o2 = spec_trace.trace[1].data['Y']

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        transform = soss_solver.solve_transform(deepframe, np.isnan(deepframe),
                                                xref_o1, yref_o1,
                                                xref_o2, yref_o2,
                                                soss_filter='SUBSTRIP256',
                                                is_fitted=(True, True, True),
                                                guess_transform=(0, 0, 0))

    if show_plots is True:
        cen_o1, cen_o2, cen_o3 = get_trace_centroids(deepframe, 'SUBSTRIP256')
        xdat_o1, ydat_o1 = cen_o1
        xdat_o2, ydat_o2 = cen_o2
        xdat_o3, ydat_o3 = cen_o3

        xtrans_o1, ytrans_o1 = soss_solver.transform_coords(*transform,
                                                            xref_o1, yref_o1)
        xtrans_o2, ytrans_o2 = soss_solver.transform_coords(*transform,
                                                            xref_o2, yref_o2)
        labels=['Extracted Centroids', 'Reference Centroids',
                'Transformed Centroids']
        plotting.do_centroid_plot(deepframe, [xdat_o1, xref_o1, xtrans_o1],
                                  [ydat_o1, yref_o1, ytrans_o1],
                                  [xdat_o2, xref_o2, xtrans_o2],
                                  [ydat_o2, yref_o2, ytrans_o2],
                                  [xdat_o3], [ydat_o3], labels=labels)

    return transform


def get_trace_centroids(deepframe, subarray):

    dimy, dimx = np.shape(deepframe)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        centroids = get_soss_centroids(deepframe, subarray=subarray)

    X1, Y1 = centroids['order 1']['X centroid'], centroids['order 1'][
        'Y centroid']
    X2, Y2 = centroids['order 2']['X centroid'], centroids['order 2'][
        'Y centroid']
    X3, Y3 = centroids['order 3']['X centroid'], centroids['order 3'][
        'Y centroid']
    ii = np.where((X1 >= 0) & (X1 <= dimx - 1))
    ii2 = np.where((X2 >= 0) & (X2 <= dimx - 1) & (Y2 <= dimy - 1))
    ii3 = np.where((X3 >= 0) & (X3 <= dimx - 1) & (Y3 <= dimy - 1))

    # Interpolate onto native pixel grid
    x1 = np.arange(dimx)
    y1 = np.interp(x1, X1[ii], Y1[ii])
    x2 = np.arange(np.max(np.floor(X2[ii2]).astype(int)))
    y2 = np.interp(x2, X2[ii2], Y2[ii2])
    x3 = np.arange(np.max(np.floor(X3[ii3]).astype(int)))
    y3 = np.interp(x3, X3[ii3], Y3[ii3])

    cen_o1, cen_o2, cen_o3 = [x1, y1], [x2, y2], [x3, y3]

    return cen_o1, cen_o2, cen_o3


