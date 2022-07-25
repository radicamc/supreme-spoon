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
from datetime import datetime
import glob
import numpy as np
import os
import pandas as pd
import warnings

from jwst import datamodels

from sys import path
applesoss_path = '/home/radica/GitHub/APPLESOSS/'
path.insert(1, applesoss_path)

from APPLESOSS.edgetrigger_centroids import get_soss_centroids


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


def make_deepstack(cube):
    """Make deepstack of a TSO.
    """
    deepstack = bn.nanmedian(cube, axis=0)
    return deepstack


def unpack_spectra(datafile, quantities=('WAVELENGTH', 'FLUX', 'FLUX_ERROR')):

    if isinstance(datafile, str):
        multi_spec = datamodels.open(datafile)
    else:
        multi_spec = datafile

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
    old_files = np.atleast_1d(old_files)
    # Hack to fix file names
    new_files = []
    for file in old_files:
        if isinstance(file, str):
            file = datamodels.open(file)

        old_filename = file.meta.filename

        split = old_filename.split(to_remove)
        new_filename = split[0] + '_' + split[1]
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


def get_trace_centroids(deepframe, subarray, output_dir=None,
                        save_results=True, save_filename=None):
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

    if save_results is True:
        yy2 = np.ones_like(x1) * np.nan
        yy2[:len(y2)] = y2
        yy3 = np.ones_like(x1) * np.nan
        yy3[:len(y3)] = y3

        centroids_dict = {'xcen o1': x1, 'ycen o1': y1,
                          'xcen o2': x1, 'ycen o2': yy2,
                          'xcen o3': x1, 'ycen o3': yy3}
        df = pd.DataFrame(data=centroids_dict)
        outfile_name = output_dir + save_filename + '_centroids.csv'
        outfile = open(outfile_name, 'a')
        outfile.write('# File Contents: Edgetrigger trace centroids\n')
        outfile.write('# File Creation Date: {}\n'.format(datetime.utcnow().replace(microsecond=0).isoformat()))
        outfile.write('# File Author: MCR\n')
        df.to_csv(outfile, index=False)
        outfile.close()
        print('Centroids saved to {}'.format(outfile_name))

    cen_o1, cen_o2, cen_o3 = [x1, y1], [x2, y2], [x3, y3]

    return cen_o1, cen_o2, cen_o3


# TODO: reformat
def pack_spectra(filename, w1, f1, e1, w2, f2, e2, t, header_dict=None,
                 header_comments=None, save_results=True):
    hdr = fits.Header()
    if header_dict is not None:
        for key in header_dict:
            hdr[key] = header_dict[key]
            if key in header_comments.keys():
                hdr.comments[key] = header_comments[key]
    hdu1 = fits.PrimaryHDU(header=hdr)
    hdr = fits.Header()
    hdr['EXTNAME'] = "Wave 2D Order 1"
    hdr['UNITS'] = "Micron"
    hdu2 = fits.ImageHDU(w1, header=hdr)
    hdr = fits.Header()
    hdr['EXTNAME'] = "Flux Order 1"
    hdr['UNITS'] = "Electrons"
    hdu3 = fits.ImageHDU(f1, header=hdr)
    hdr = fits.Header()
    hdr['EXTNAME'] = "Flux Error Order 1"
    hdr['UNITS'] = "Electrons"
    hdu4 = fits.ImageHDU(e1, header=hdr)
    hdr = fits.Header()
    hdr['EXTNAME'] = "Wave 2D Order 2"
    hdr['UNITS'] = "Micron"
    hdu5 = fits.ImageHDU(w2, header=hdr)
    hdr = fits.Header()
    hdr['EXTNAME'] = "Flux Order 2"
    hdr['UNITS'] = "Electrons"
    hdu6 = fits.ImageHDU(f2, header=hdr)
    hdr = fits.Header()
    hdr['EXTNAME'] = "Flux Error Order 2"
    hdr['UNITS'] = "Electrons"
    hdu7 = fits.ImageHDU(e2, header=hdr)
    hdr = fits.Header()
    hdr['EXTNAME'] = "Time"
    hdr['UNITS'] = "BJD"
    hdu8 = fits.ImageHDU(t, header=hdr)
    if save_results is True:
        hdul = fits.HDUList([hdu1, hdu2, hdu3, hdu4, hdu5, hdu6, hdu7, hdu8])
        hdul.writeto(filename, overwrite=True)

    param_dict = {'Wave 2D Order 1': w1, 'Flux Order 1': f1,
                  'Flux Error Order 1': e1, 'Wave 2D Order 2': w2,
                  'Flux Order 2': f2, 'Flux Error Order 2': e2,
                  'Time': t}

    return param_dict


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


def get_default_header():
    header_dict = {'Target_Name': None,
                   'Instrument': 'NIRISS/SOSS',
                   'Date': datetime.utcnow().replace(microsecond=0).isoformat(),
                   'Pipeline': 'Supreme Spoon',
                   'Author': 'MCR',
                   'Contents': None,
                   'Extraction Method': 'Box Extraction'}
    header_comments = {'Target_Name': 'Name of the target',
                       'Instrument': 'Instrument used to acquire the data',
                       'Date': 'UTC date file created',
                       'Pipeline': 'Pipeline that produced this file',
                       'Author': 'File author',
                       'Contents': 'Description of file contents',
                       'Extraction Method': 'Type of 1D extraction'}

    return header_dict, header_comments


def get_dn2e(datafile):
    if isinstance(datafile, str):
        data = datamodels.open(datafile)
    else:
        data = datafile
    ngroup = data.meta.exposure.ngroups
    frame_time = data.meta.exposure.frame_time
    gain_factor = 1.6
    dn2e = gain_factor * (ngroup - 1) * frame_time

    return dn2e


def unpack_input_directory(indir, filetag='', process_f277w=False):
    if indir[-1] != '/':
        indir += '/'
    all_files = glob.glob(indir + '*')
    clear_segments = []
    f277w_segments = []
    for file in all_files:
        try:
            header = fits.getheader(file, 0)
        except(OSError, IsADirectoryError):
            continue
        if header['FILTER'] == 'CLEAR':
            if filetag in file:
                clear_segments.append(file)
        elif header['FILTER'] == 'F277W' and process_f277w is True:
            if filetag in file:
                f277w_segments.append(file)
        else:
            continue
    # Ensure that segments are packed in chronological order
    if len(clear_segments) > 1:
        clear_segments = np.array(clear_segments)
        segment_numbers = []
        for file in clear_segments:
            seg_no = fits.getheader(file, 0)['EXSEGNUM']
            segment_numbers.append(seg_no)
        correct_order = np.argsort(segment_numbers)
        clear_segments = clear_segments[correct_order]
    if len(f277w_segments) > 1:
        f277w_segments = np.array(f277w_segments)
        segment_numbers = []
        for file in f277w_segments:
            seg_no = fits.getheader(file, 0)['EXSEGNUM']
            segment_numbers.append(seg_no)
        correct_order = np.argsort(segment_numbers)
        f277w_segments = f277w_segments[correct_order]

    return clear_segments, f277w_segments
