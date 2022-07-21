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
import numpy as np
import os
import warnings

from jwst import datamodels
from jwst.extract_1d.soss_extract import soss_solver
from jwst.pipeline import calwebb_spec2

from sys import path
applesoss_path = '/users/michaelradica/Documents/GitHub/APPLESOSS/'
path.insert(1, applesoss_path)

from APPLESOSS.edgetrigger_centroids import get_soss_centroids

from supreme_spoon import plotting


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


def determine_soss_transform(deepframe, datafile, show_plots=False):

    step = calwebb_spec2.extract_1d_step.Extract1dStep()
    spectrace_ref = step.get_reference_file(datafile, 'spectrace')
    spec_trace = datamodels.SpecTraceModel(spectrace_ref)

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
    # TODO: save output
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


def write_spectra_to_file(filename, w1, f1, e1, w2, f2, e2, t,
                          header_dict=None, header_comments=None):
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
    hdul = fits.HDUList([hdu1, hdu2, hdu3, hdu4, hdu5, hdu6, hdu7, hdu8])
    hdul.writeto(filename, overwrite=True)


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
                   'Pipeline': 'Supreme Spoon',
                   'Date': datetime.utcnow().replace(microsecond=0).isoformat(),
                   'Author': 'MCR',
                   'Contents': 'Full resolution 1D stellar spectra'}
    header_comments = {'Target_Name': 'Name of the target',
                       'Instrument': 'Instrument used to acquire the data',
                       'Pipeline': 'Pipeline that produced this file',
                       'Date': 'UTC date file created',
                       'Author': 'File author',
                       'Contents': 'Description of file contents'}

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
