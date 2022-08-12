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
import juliet
import numpy as np
import os
import pandas as pd
import pickle as pk
import warnings

from jwst import datamodels

from sys import path
applesoss_path = '/home/radica/GitHub/APPLESOSS/'
path.insert(1, applesoss_path)
try:
    from APPLESOSS.edgetrigger_centroids import get_soss_centroids
    use_applesoss = True
except ModuleNotFoundError:
    msg = 'APPLESOSS module not available. Some capabilities will be limited.'
    warnings.warn(msg)
    use_applesoss = False


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


def do_replacement(frame, badpix_map, dq=None, box_size=5):
    """Replace flagged pixels with the median of a surrounding box.
    """

    dimy, dimx = np.shape(frame)
    frame_out = np.copy(frame)
    if dq is not None:
        dq_out = np.copy(dq)
    else:
        dq_out = np.zeros_like(frame)

    # Loop over all flagged pixels.
    for i in range(dimx):
        for j in range(dimy):
            if badpix_map[j, i] == 0:
                continue
            # If pixel is flagged, replace it with the box median.
            else:
                med = get_interp_box(frame, box_size, i, j, dimx, dimy)[0]
                frame_out[j, i] = med
                dq_out[j, i] = 0

    return frame_out, dq_out


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
    t_start = Time(t_start, format='isot', scale='tdb')
    t = np.arange(nint) * tgroup * ngroup + t_start.mjd
    return t


def fix_filenames(old_files, to_remove, outdir, to_add=''):
    old_files = np.atleast_1d(old_files)
    # Hack to fix file names
    new_files = []
    for file in old_files:
        if isinstance(file, str):
            file = datamodels.open(file)

        old_filename = file.meta.filename

        split = old_filename.split(to_remove)
        new_filename = split[0] + '_' + split[1]
        if to_add != '':
            temp = new_filename.split('.fits')
            new_filename = temp[0] + '_' + to_add + '.fits'
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

    if use_applesoss is False:
        msg = 'APPLESOSS module not available.\n Cannot extract trace centroids.'
        raise ModuleNotFoundError(msg)
    dimy, dimx = np.shape(deepframe)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        centroids = get_soss_centroids(deepframe, subarray=subarray)

    X1, Y1 = centroids['order 1']['X centroid'], centroids['order 1']['Y centroid']
    X2, Y2 = centroids['order 2']['X centroid'], centroids['order 2']['Y centroid']
    X3, Y3 = centroids['order 3']['X centroid'], centroids['order 3']['Y centroid']
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
        if save_filename[-1] != '_':
            save_filename += '_'
        outfile_name = output_dir + save_filename + 'centroids.csv'
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
def pack_spectra(filename, wl1, wu1, f1, e1, wl2, wu2, f2, e2, t,
                 header_dict=None, header_comments=None, save_results=True):

    hdr = fits.Header()
    if header_dict is not None:
        for key in header_dict:
            hdr[key] = header_dict[key]
            if key in header_comments.keys():
                hdr.comments[key] = header_comments[key]
    hdu1 = fits.PrimaryHDU(header=hdr)

    hdr = fits.Header()
    hdr['EXTNAME'] = "Wave Low O1"
    hdr['UNITS'] = "Micron"
    hdu2 = fits.ImageHDU(wl1, header=hdr)

    hdr = fits.Header()
    hdr['EXTNAME'] = "Wave Up O1"
    hdr['UNITS'] = "Micron"
    hdu3 = fits.ImageHDU(wu1, header=hdr)

    hdr = fits.Header()
    hdr['EXTNAME'] = "Flux O1"
    hdr['UNITS'] = "Electrons"
    hdu4 = fits.ImageHDU(f1, header=hdr)

    hdr = fits.Header()
    hdr['EXTNAME'] = "Flux Err O1"
    hdr['UNITS'] = "Electrons"
    hdu5 = fits.ImageHDU(e1, header=hdr)

    hdr = fits.Header()
    hdr['EXTNAME'] = "Wave Low O2"
    hdr['UNITS'] = "Micron"
    hdu6 = fits.ImageHDU(wl2, header=hdr)

    hdr = fits.Header()
    hdr['EXTNAME'] = "Wave Up O2"
    hdr['UNITS'] = "Micron"
    hdu7 = fits.ImageHDU(wu2, header=hdr)

    hdr = fits.Header()
    hdr['EXTNAME'] = "Flux O2"
    hdr['UNITS'] = "Electrons"
    hdu8 = fits.ImageHDU(f2, header=hdr)

    hdr = fits.Header()
    hdr['EXTNAME'] = "Flux Err O2"
    hdr['UNITS'] = "Electrons"
    hdu9 = fits.ImageHDU(e2, header=hdr)

    hdr = fits.Header()
    hdr['EXTNAME'] = "Time"
    hdr['UNITS'] = "BJD"
    hdu10 = fits.ImageHDU(t, header=hdr)

    if save_results is True:
        hdul = fits.HDUList([hdu1, hdu2, hdu3, hdu4, hdu5, hdu6, hdu7, hdu8,
                             hdu9, hdu10])
        hdul.writeto(filename, overwrite=True)

    param_dict = {'Wave Low O1': wl1, 'Wave Up O1': wu1, 'Flux O1': f1,
                  'Flux Err O1': e1, 'Wave Low O2': wl2, 'Wave UP O2': wu2,
                  'Flux O2': f2, 'Flux Err O2': e2, 'Time': t}

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
    header_dict = {'Target': None,
                   'Inst': 'NIRISS/SOSS',
                   'Date': datetime.utcnow().replace(microsecond=0).isoformat(),
                   'Pipeline': 'Supreme-SPOON',
                   'Author': 'MCR',
                   'Contents': None,
                   'Method': 'Box Extraction',
                   'Width': 25,
                   'Transx': 0,
                   'Transy': 0,
                   'Transth': 0}
    header_comments = {'Target': 'Name of the target',
                       'Inst': 'Instrument used to acquire the data',
                       'Date': 'UTC date file created',
                       'Pipeline': 'Pipeline that produced this file',
                       'Author': 'File author',
                       'Contents': 'Description of file contents',
                       'Method': 'Type of 1D extraction',
                       'Width': 'Box width',
                       'Transx': 'SOSS transform dx',
                       'Transy': 'SOSS transform dy',
                       'Transth': 'SOSS transform dtheta'}

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


def unpack_input_directory(indir, filetag='', exposure_type='CLEAR'):
    if indir[-1] != '/':
        indir += '/'
    all_files = glob.glob(indir + '*')
    segments = []
    for file in all_files:
        try:
            header = fits.getheader(file, 0)
        except(OSError, IsADirectoryError):
            continue
        try:
            if header['FILTER'] == exposure_type:
                if filetag in file:
                    segments.append(file)
            else:
                continue
        except KeyError:
            continue
    # Ensure that segments are packed in chronological order
    if len(segments) > 1:
        segments = np.array(segments)
        segment_numbers = []
        for file in segments:
            seg_no = fits.getheader(file, 0)['EXSEGNUM']
            segment_numbers.append(seg_no)
        correct_order = np.argsort(segment_numbers)
        segments = segments[correct_order]

    return segments


def remove_nans(datamodel):
    if isinstance(datamodel, str):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            datamodel = datamodels.open(datamodel)
    modelout = datamodel.copy()
    ind = (~np.isfinite(datamodel.data)) | (~np.isfinite(datamodel.err))
    modelout.data[ind] = 0
    modelout.err[ind] = np.nanmedian(datamodel.err) * 10
    modelout.dq[ind] += 1
    return modelout


def get_soss_estimate(atoca_spectra, output_dir):
    atoca_spec = datamodels.open(atoca_spectra)

    for spec in atoca_spec.spec:
        if spec.meta.soss_extract1d.type == 'OBSERVATION':
            estimate = datamodels.SpecModel(spec_table=spec.spec_table)
            break
    estimate_filename = estimate.save(output_dir + 'soss_estimate.fits')

    return estimate_filename


def gen_ldprior_from_nestor(filename, wave):
    aa = open(filename, "rb")
    nestor_priors = pk.load(aa)
    aa.close()

    lw = np.concatenate([wave[:, None], np.roll(wave, 1)[:, None]], axis=1)
    up = np.concatenate([wave[:, None], np.roll(wave, -1)[:, None]], axis=1)

    uperr = (np.mean(up, axis=1) - wave)[:-1]
    uperr = np.insert(uperr, -1, uperr[-1])

    lwerr = (wave - np.mean(lw, axis=1))[1:]
    lwerr = np.insert(lwerr, 0, lwerr[0])
    bins_low = wave - lwerr
    bins_up = wave + uperr

    prior_q1, prior_q2 = [], []
    for bin_low, bin_up in zip(bins_low, bins_up):
        current_q1, current_q2 = [], []
        for key in nestor_priors.keys():
            nwave = float(key.split('_')[-1])
            if nwave >= bin_low and nwave < bin_up:
                current_q1.append(
                    nestor_priors[key]['q1_SOSS']['hyperparameters'][0])
                current_q2.append(
                    nestor_priors[key]['q2_SOSS']['hyperparameters'][0])

        current_q1 = np.array(current_q1)
        current_q2 = np.array(current_q2)
        prior_q1.append(np.mean(current_q1))
        prior_q2.append(np.mean(current_q2))

    return prior_q1, prior_q2


def get_wavebin_limits(wave):
    up = np.concatenate([wave[:, None], np.roll(wave, 1)[:, None]], axis=1)
    low = np.concatenate([wave[:, None], np.roll(wave, -1)[:, None]], axis=1)

    bin_low = (np.mean(low, axis=1))[:-1]
    bin_low = np.insert(bin_low, -1, bin_low[-1])

    bin_up = (np.mean(up, axis=1))[1:]
    bin_up = np.insert(bin_up, 0, bin_up[0])

    return bin_low, bin_up


def open_filetype(datafile):
    if isinstance(datafile, str):
        data = datamodels.open(datafile)
    elif isinstance(datafile, (datamodels.CubeModel, datamodels.RampModel)):
        data = datafile
    else:
        raise ValueError('Invalid filetype: {}'.format(type(datafile)))
    return data


def get_ld_prior(order, wavebin_low, wavebin_up):
    rawprior = pd.read_csv('w96_order{}_lds_spam_quadratic.txt'.format(order),
                           sep=' ',
                           comment='#', usecols=[0, 7, 8],
                           names=['wave', 'u1', 'u2'])
    q1, q2 = juliet.reverse_q_coeffs('quadratic', rawprior['u1'].values,
                                     rawprior['u2'].values)

    prior_q1, prior_q2 = [], []
    for wl, wu in zip(wavebin_low, wavebin_up):
        current_q1, current_q2 = [], []
        for i, w in enumerate(rawprior['wave'].values):
            if w > wl and w <= wu:
                current_q1.append(q1[i])
                current_q2.append(q2[i])
            elif w > wu:
                prior_q1.append(np.nanmean(current_q1))
                prior_q2.append(np.nanmean(current_q2))
                break

    return prior_q1, prior_q2


def outlier_resistant_variance(data):
    var = (np.nanmedian(np.abs(data-np.nanmedian(data)))/0.6745)**2
    return var


def get_ld_coefs(filename, wavebin_low, wavebin_up):
    ld = pd.read_csv(filename, comment='#', sep=',')
    q1s, q2s = juliet.reverse_q_coeffs('quadratic', ld['c1'].values, ld['c2'].values)
    waves = ld['wave'].values
    ii = np.argsort(waves)
    waves = waves[ii]
    q1s = q1s[ii]
    q2s = q2s[ii]

    prior_q1, prior_q2 = [], []
    for wl, wu in zip(wavebin_low, wavebin_up):
        current_q1, current_q2 = [], []
        for w, q1, q2 in zip(waves, q1s, q2s):
            if w > wl and w <= wu:
                current_q1.append(q1)
                current_q2.append(q2)
            elif w > wu:
                prior_q1.append(np.nanmean(current_q1))
                prior_q2.append(np.nanmean(current_q2))
                break
    return prior_q1, prior_q2


def package_ld_priors(wave, c1, c2, order, target, M_H, Teff, logg, outdir):
    dd = {'wave': wave,
          'c1': c1,
          'c2': c2}
    df = pd.DataFrame(data=dd)
    filename = target+'_order' + str(order) + '_exotic-ld_quadratic.csv'
    if os.path.exists(outdir + filename):
        os.remove(outdir + filename)
    f = open(outdir + filename, 'a')
    f.write('# Target: {}\n'.format(target))
    f.write('# Instrument: NIRISS/SOSS\n')
    f.write('# Order: {}\n'.format(order))
    f.write('# Author: {}\n'.format(os.environ.get('USER')))
    f.write('# Date: {}\n'.format(datetime.utcnow().replace(microsecond=0).isoformat()))
    f.write('# Stellar M/H: {}\n'.format(M_H))
    f.write('# Stellar log g: {}\n'.format(logg))
    f.write('# Stellar Teff: {}\n'.format(Teff))
    f.write('# Algorithm: ExoTiC-LD\n')
    f.write('# Limb Darkening Model: quadratic\n')
    f.write('# Column wave: Central wavelength of bin (micron)\n')
    f.write('# Column c1: Quadratic Coefficient 1\n')
    f.write('# Column c2: Quadratic Coefficient 2\n')
    f.write('#\n')
    df.to_csv(f, index=False)
    f.close()


def get_filename_root(datafiles):
    fileroots = []
    for file in datafiles:
        if isinstance(file, str):
            data = datamodels.open(file)
        else:
            data = file
        filename_split = data.meta.filename.split('_')
        fileroot = ''
        for chunk in filename_split[:-1]:
            fileroot += chunk + '_'
        fileroots.append(fileroot)
    return fileroots
