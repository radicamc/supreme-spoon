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

from jwst import datamodels


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




