#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 08:51 2022

@author: MCR

Miscellaneous pipeline tools.
"""

from astropy.io import fits
import bottleneck as bn
from datetime import datetime
import glob
import juliet
import numpy as np
import os
import pandas as pd
from scipy.interpolate import interp2d
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from tqdm import tqdm
import warnings
import yaml

from applesoss.edgetrigger_centroids import get_soss_centroids
from jwst import datamodels


def do_replacement(frame, badpix_map, dq=None, box_size=5):
    """Replace flagged pixels with the median of a surrounding box.

    Parameters
    ----------
    frame : array-like[float]
        Data frame.
    badpix_map : array-like[bool]
        Map of pixels to be replaced.
    dq : array-like[int]
        Data quality flags.
    box_size : int
        Size of box to consider.

    Returns
    -------
    frame_out : array-like[float]
        Input frame wth pixels interpolated.
    dq_out : array-like[int]
        Input dq map with interpolated pixels set to zero.
    """

    dimy, dimx = np.shape(frame)
    frame_out = np.copy(frame)
    # Get the data quality flags.
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
                med = get_interp_box(frame, box_size, i, j, dimx)[0]
                frame_out[j, i] = med
                # Set dq flag of inerpolated pixel to zero (use the pixel).
                dq_out[j, i] = 0

    return frame_out, dq_out


def fix_filenames(old_files, to_remove, outdir, to_add=''):
    """Hacky function to remove file extensions that get added when running a
    default JWST DMS step after a custom one.

    Parameters
    ----------
    old_files : array-like[str], array-like[jwst.datamodel]
        List of datamodels or paths to datamodels.
    to_remove : str
        File name extension to be removed.
    outdir : str
        Directory to which to save results.
    to_add : str
        Extention to add to the file name.

    Returns
    -------
    new_files : array-like[str]
        New file names.
    """

    old_files = np.atleast_1d(old_files)
    new_files = []
    # Open datamodel and get file name.
    for file in old_files:
        if isinstance(file, str):
            file = datamodels.open(file)
        old_filename = file.meta.filename

        # Remove the unwanted extention.
        split = old_filename.split(to_remove)
        new_filename = split[0] + '_' + split[1]
        # Add extension if necessary.
        if to_add != '':
            temp = new_filename.split('.fits')
            new_filename = temp[0] + '_' + to_add + '.fits'

        # Save file with new filename
        file.write(outdir + new_filename)
        new_files.append(outdir + new_filename)
        file.close()

        # Get rid of old file.
        os.remove(outdir + old_filename)

    return new_files


def format_out_frames(out_frames, occultation_type='transit'):
    """Create a mask of baseline flux frames for lightcurve normalization.
    Either out-of-transit integrations for transits or in-eclipse integrations
    for eclipses.

    Parameters
    ----------
    out_frames : array-like[int]
        Integration numbers of ingress and egress.
    occultation_type : str
        Type of occultation, either 'transit' or 'eclipse'.

    Returns
    -------
    baseline_ints : array-like[int]
        Array of out-of-transit, or in-eclipse frames for transits and
        eclipses respectively.

    Raises
    ------
    ValueError
        If an unknown occultation type is passed.
    """

    if occultation_type == 'transit':
        # Format the out-of-transit integration numbers.
        out_frames = np.abs(out_frames)
        out_frames = np.concatenate([np.arange(out_frames[0]),
                                     np.arange(out_frames[1]) - out_frames[1]])
    elif occultation_type == 'eclipse':
        # Format the in-eclpse integration numbers.
        out_frames = np.linspace(out_frames[0], out_frames[1],
                                 out_frames[1] - out_frames[0] + 1).astype(int)
    else:
        msg = 'Unknown Occultaton Type: {}'.format(occultation_type)
        raise ValueError(msg)

    return out_frames


def get_dn2e(datafile):
    """Determine the correct DN/s to e- conversion based on the integration
    time and estimated gain.

    Parameters
    ----------
    datafile : str, jwst.datamodel
        Path to datamodel, or datamodel itself.

    Returns
    -------
    dn2e : float
        DN/s to e- conversion factor.
    """

    data = open_filetype(datafile)
    # Get number of groups and group time (each group is one frame).
    ngroup = data.meta.exposure.ngroups
    frame_time = data.meta.exposure.frame_time
    # Approximate gain factor. Gain varies across the detector, but is ~1.6
    # on average.
    gain_factor = 1.6
    # Calculate the DN/s to e- conversion by multiplying by integration time
    # and gain factor. Note that the integration time uses ngroup-1, due to
    # up-the-ramp fitting.
    dn2e = gain_factor * (ngroup - 1) * frame_time

    return dn2e


def get_default_header():
    """Format the default header for the lightcurve file.

    Returns
    -------
    header_dict : dict
        Header keyword dictionary.
    header_commets : dict
        Header comment dictionary.
    """

    # Header with important keywords.
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
    # Explanations of keywords.
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


def get_filename_root(datafiles):
    """Get the file name roots for each segment.

    Parameters
    ----------
    datafiles : array-like[str], array-like[jwst.datamodel]
        Datamodels, or paths to datamodels for each segment.

    Returns
    -------
    fileroots : array-like[str]
        List of file name roots.
    """

    fileroots = []
    for file in datafiles:
        # Open the datamodel.
        if isinstance(file, str):
            data = datamodels.open(file)
        else:
            data = file
        # Get the last part of the path, and split file name into chunks.
        filename_split = data.meta.filename.split('/')[-1].split('_')
        fileroot = ''
        # Get the filename before the step info and save.
        for chunk in filename_split[:-1]:
            fileroot += chunk + '_'
        fileroots.append(fileroot)

    return fileroots


def get_filename_root_noseg(fileroots):
    """Get the file name root for a SOSS TSO woth noo segment information.

    Parameters
    ----------
    fileroots : array-like[str]
        File root names for each segment.

    Returns
    -------
    fileroot_noseg : str
        File name root with no segment information.
    """

    # Get total file root, with no segment info.
    working_name = fileroots[0]
    if 'seg' in working_name:
        parts = working_name.split('seg')
        part1, part2 = parts[0][:-1], parts[1][3:]
        fileroot_noseg = part1 + part2
    else:
        fileroot_noseg = fileroots[0]

    return fileroot_noseg


def get_interp_box(data, box_size, i, j, dimx):
    """Get median and standard deviation of a box centered on a specified
    pixel.

    Parameters
    ----------
    data : array-like[float]
        Data frame.
    box_size : int
        Size of box to consider.
    i : int
        X pixel.
    j : int
        Y pixel.
    dimx : int
        Size of x dimension.

    Returns
    -------
    box_properties : array-like
        Median and standard deviation of pixels in the box.
    """

    # Get the box limits.
    low_x = np.max([i - box_size, 0])
    up_x = np.min([i + box_size, dimx - 1])

    # Calculate median and std deviation of box.
    median = np.nanmedian(data[j, low_x:up_x])
    stddev = np.nanstd(data[j, low_x:up_x])

    # Pack into array.
    box_properties = np.array([median, stddev])

    return box_properties


def get_soss_estimate(atoca_spectra, output_dir):
    """Convert the AtocaSpectra output of ATOCA into the format expected for a
    soss_estimate.

    Parameters
    ----------
    atoca_spectra : str, MultiSpecModel
        AtocaSpectra datamodel, or path to the datamodel.
    output_dir : str
        Directory to which to save results.

    Returns
    -------
    estimate_filename : str
        Path to soss_estimate file.
    """

    # Open the AtocaSpectra file.
    atoca_spec = datamodels.open(atoca_spectra)
    # Get the spectrum.
    for spec in atoca_spec.spec:
        if spec.meta.soss_extract1d.type == 'OBSERVATION':
            estimate = datamodels.SpecModel(spec_table=spec.spec_table)
            break
    # Save the spectrum as a soss_estimate file.
    estimate_filename = estimate.save(output_dir + 'soss_estimate.fits')

    return estimate_filename


def get_timestamps(datafiles):
    """Get the mid-time stamp for each integration in BJD,

    Parameters
    ----------
    datafiles : array-like[jwst.datamodel], jwst.datamodel
        Datamodels for each segment in a TSO.

    Returns
    -------
    times : array-like[float]
        Mid-integration times for each integraton in BJD.
    """

    datafiles = np.atleast_1d(datafiles)
    # Loop over all data files and get mid integration time stamps.
    for i, data in enumerate(datafiles):
        data = datamodels.open(data)
        if i == 0:
            times = data.int_times['int_mid_BJD_TDB']
        else:
            times = np.concatenate([times, data.int_times['int_mid_BJD_TDB']])

    return times


def get_trace_centroids(deepframe, tracetable, subarray, save_results=True,
                        save_filename=''):
    """Get the trace centroids for all three orders via the edgetrigger method.

    Parameters
    ----------
    deepframe : array-like[float]
        Median stack.
    tracetable : str
        Path to SpecTrace reference file.
    subarray : str
        Subarray identifier.
    save_results : bool
        If True, save results to file.
    save_filename : str
        Filename of save file.

    Returns
    -------
    cen_o1 : array-like[float]
        Order 1 X and Y centroids.
    cen_o2 : array-like[float]
        Order 2 X and Y centroids.
    cen_o3 : array-like[float]
        Order 3 X and Y centroids.
    """

    dimy, dimx = np.shape(deepframe)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        cens = get_soss_centroids(deepframe, tracetable,
                                  subarray=subarray)

    x1, y1 = cens['order 1']['X centroid'], cens['order 1']['Y centroid']
    ii = np.where((x1 >= 0) & (y1 <= dimx - 1))
    # Interpolate onto native pixel grid
    xx1 = np.arange(dimx)
    yy1 = np.interp(xx1, x1[ii], y1[ii])

    if subarray != 'SUBSTRIP96':
        x2, y2 = cens['order 2']['X centroid'], cens['order 2']['Y centroid']
        x3, y3 = cens['order 3']['X centroid'], cens['order 3']['Y centroid']
        ii2 = np.where((x2 >= 0) & (x2 <= dimx - 1) & (y2 <= dimy - 1))
        ii3 = np.where((x3 >= 0) & (x3 <= dimx - 1) & (y3 <= dimy - 1))
        # Interpolate onto native pixel grid
        xx2 = np.arange(np.max(np.floor(x2[ii2]).astype(int)))
        yy2 = np.interp(xx2, x2[ii2], y2[ii2])
        xx3 = np.arange(np.max(np.floor(x3[ii3]).astype(int)))
        yy3 = np.interp(xx3, x3[ii3], y3[ii3])
    else:
        xx2, yy2 = xx1, np.ones_like(xx1) * np.nan
        xx3, yy3 = xx1, np.ones_like(xx1) * np.nan

    if save_results is True:
        yyy2 = np.ones_like(xx1) * np.nan
        yyy2[:len(yy2)] = yy2
        yyy3 = np.ones_like(xx1) * np.nan
        yyy3[:len(yy3)] = yy3

        centroids_dict = {'xpos': xx1, 'ypos o1': yy1, 'ypos o2': yyy2,
                          'ypos o3': yyy3}
        df = pd.DataFrame(data=centroids_dict)
        if save_filename[-1] != '_':
            save_filename += '_'
        outfile_name = save_filename + 'centroids.csv'
        outfile = open(outfile_name, 'a')
        outfile.write('# File Contents: Edgetrigger trace centroids\n')
        outfile.write('# File Creation Date: {}\n'.format(
            datetime.utcnow().replace(microsecond=0).isoformat()))
        outfile.write('# File Author: MCR\n')
        df.to_csv(outfile, index=False)
        outfile.close()
        print('Centroids saved to {}'.format(outfile_name))

    cen_o1 = np.array([xx1, yy1])
    cen_o2 = np.array([xx2, yy2])
    cen_o3 = np.array([xx3, yy3])

    return cen_o1, cen_o2, cen_o3


def get_wavebin_limits(wave):
    """Determine the upper and lower limits of wavelength bins centered on a
    given wavelength axis.

    Parameters
    ----------
    wave : array-like[float]
        Wavelengh array.

    Returns
    -------
    bin_low : array-like[float]
        Lower edge of wavelength bin.
    bin_up : array-like[float]
        Upper edge of wavelength bin.
    """

    # Shift wavelength array by one element forward and backwards, and create
    # 2D stack where each wavelength is sandwiched between its upper or lower
    # neighbour respectively.
    up = np.concatenate([wave[:, None], np.roll(wave, 1)[:, None]], axis=1)
    low = np.concatenate([wave[:, None], np.roll(wave, -1)[:, None]], axis=1)

    # Take the mean in the vertical direction to get the midpoint between the
    # two wavelengths. Use this as the bin limits.
    bin_low = (np.mean(low, axis=1))[:-1]
    bin_low = np.insert(bin_low, -1, bin_low[-1])
    bin_up = (np.mean(up, axis=1))[1:]
    bin_up = np.insert(bin_up, 0, bin_up[0])

    return bin_low, bin_up


def make_deepstack(cube):
    """Make deep stack of a TSO.

    Parameters
    ----------
    cube : array-like[float]
        Stack of all integrations in a TSO

    Returns
    -------
    deepstack : array-like[float]
       Median of the input cube along the integration axis.
    """

    # Take median of input cube along the integration axis.
    deepstack = bn.nanmedian(cube, axis=0)

    return deepstack


def open_filetype(datafile):
    """Open a datamodel whether it is a path, or the datamodel itself.

    Parameters
    ----------
    datafile : str, jwst.datamodel
        Datamodel or path to datamodel.

    Returns
    -------
    data : jwst.datamodel
        Opened datamodel.

    Raises
    ------
    ValueError
        If the filetype passed is not str or jwst.datamodel.
    """

    if isinstance(datafile, str):
        data = datamodels.open(datafile)
    elif isinstance(datafile, (datamodels.CubeModel, datamodels.RampModel,
                               datamodels.MultiSpecModel)):
        data = datafile
    else:
        raise ValueError('Invalid filetype: {}'.format(type(datafile)))

    return data


def open_stage2_secondary_outputs(deep_file, centroid_file, smoothed_wlc_file,
                                  output_tag='', root_dir='./'):
    """Utlity to locate and read in secondary outputs from stage 2.

    Parameters
    ----------
    deep_file : str, None
        Path to deep frame file.
    centroid_file : str, None
        Path to centroids file.
    smoothed_wlc_file : str, None
        Path to smoothed wlc scaling file.
    root_dir : str
        Root directory.
    output_tag : str
        Tag given to pipeline_outputs_directory.

    Returns
    -------
    deepframe : array-like[float]
        Deep frame.
    centroids : array-like[float]
        Centroids foor all orders.
    smoothed_wlc : array-like[float]
        Smoothed wlc scaling.
    """

    input_dir = root_dir + 'pipeline_outputs_directory{}/Stage2/'.format(output_tag)
    # Locate and read in the deepframe.
    if deep_file is None:
        deep_file = glob.glob(input_dir + '*deepframe*')
        if len(deep_file) > 1:
            msg = 'Multiple deep frame files detected.'
            raise ValueError(msg)
        elif len(deep_file) == 0:
            msg = 'No deep frame file found.'
            raise FileNotFoundError(msg)
        else:
            deep_file = deep_file[0]
    deepframe = fits.getdata(deep_file)

    # Locate and read in the centroids.
    if centroid_file is None:
        centroid_file = glob.glob(input_dir + '*centroids*')
        if len(centroid_file) > 1:
            msg = 'Multiple centroid files detected.'
            raise ValueError(msg)
        elif len(centroid_file) == 0:
            msg = 'No centroid file found.'
            raise FileNotFoundError(msg)
        else:
            centroid_file = centroid_file[0]
    centroids = pd.read_csv(centroid_file, comment='#')

    # Locate and read in the smoothed wlc.
    if smoothed_wlc_file is None:
        smoothed_wlc_file = glob.glob(input_dir + '*lcestimate*')
        if len(smoothed_wlc_file) > 1:
            msg = 'Multiple WLC scaling files detected.'
            raise ValueError(msg)
        elif len(smoothed_wlc_file) == 0:
            msg = 'No WLC scaling file found.'
            raise FileNotFoundError(msg)
        else:
            smoothed_wlc_file = smoothed_wlc_file[0]
    smoothed_wlc = np.load(smoothed_wlc_file)

    return deepframe, centroids, smoothed_wlc


def outlier_resistant_variance(data):
    """Calculate the varaince of some data in an outlier resistant manner.
    """

    var = (np.nanmedian(np.abs(data - np.nanmedian(data))) / 0.6745) ** 2
    return var


def pack_ld_priors(wave, c1, c2, order, target, m_h, teff, logg, outdir):
    """Write model limb darkening parameters to a file to be used as priors
    for light curve fitting.

    Parameters
    ----------
    wave : array-like[float]
        Wavelength axis.
    c1 : array-like[float]
        c1 parameter for 2-parameter limb darkening law.
    c2 : array-like[float]
        c2 parameter for 2-parameter limb darkening law.
    order : int
        SOSS order.
    target : str
        Name of the target.
    m_h : float
        Host star metallicity.
    teff : float
        Host star effective temperature.
    logg : float
        Host star gravity.
    outdir : str
        Directory to which to save file.
    """

    # Create dictionary with model LD info.
    dd = {'wave': wave, 'c1': c1,  'c2': c2}
    df = pd.DataFrame(data=dd)
    # Remove old LD file if one exists.
    filename = target+'_order' + str(order) + '_exotic-ld_quadratic.csv'
    if os.path.exists(outdir + filename):
        os.remove(outdir + filename)
    # Add header info.
    f = open(outdir + filename, 'a')
    f.write('# Target: {}\n'.format(target))
    f.write('# Instrument: NIRISS/SOSS\n')
    f.write('# Order: {}\n'.format(order))
    f.write('# Author: {}\n'.format(os.environ.get('USER')))
    f.write('# Date: {}\n'.format(datetime.utcnow().replace(microsecond=0).isoformat()))
    f.write('# Stellar M/H: {}\n'.format(m_h))
    f.write('# Stellar log g: {}\n'.format(logg))
    f.write('# Stellar Teff: {}\n'.format(teff))
    f.write('# Algorithm: ExoTiC-LD\n')
    f.write('# Limb Darkening Model: quadratic\n')
    f.write('# Column wave: Central wavelength of bin (micron)\n')
    f.write('# Column c1: Quadratic Coefficient 1\n')
    f.write('# Column c2: Quadratic Coefficient 2\n')
    f.write('#\n')
    df.to_csv(f, index=False)
    f.close()


def pack_spectra(filename, wl1, wu1, f1, e1, wl2, wu2, f2, e2, t,
                 header_dict=None, header_comments=None, save_results=True):
    """Pack stellar spectra into a fits file.

    Parameters
    ----------
    filename : str
        File to which to save results.
    wl1 : array-like[float]
        Order 1 wavelength bin lower limits.
    wu1 : array-like[float]
        Order 1 wavelength bin upper limits.
    f1 : array-like[float]
        Order 1 flux.
    e1 : array-like[float]
        Order 1 flux error.
    wl2 : array-like[float]
        Order 2 wavelength bin lower limits.
    wu2 : array-like[float]
        Order 2 wavelength bin upper limits.
    f2 : array-like[float]
        Order 2 flux.
    e2 : array-like[float]
        Order 2 flux error.
    t : array-like[float]
        Time axis.
    header_dict : dict
        Header keywords and values.
    header_comments : dict
        Header comments.
    save_results : bool
        If True, save results to file.

    Returns
    -------
    param_dict : dict
        Lightcurve parameters packed into a dictionary.
    """

    # Initialize the fits header.
    hdr = fits.Header()
    if header_dict is not None:
        for key in header_dict:
            hdr[key] = header_dict[key]
            if key in header_comments.keys():
                hdr.comments[key] = header_comments[key]
    hdu1 = fits.PrimaryHDU(header=hdr)

    # Pack order 1 values.
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

    # Pack order 2 values.
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

    # Pack time axis.
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


def parse_config(config_file):
    """Parse a yaml config file.

    Parameters
    ----------
    config_file : str
        Path to config file.

    Returns
    -------
    config : dict
        Dictionary of config parameters.
    """

    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for key in config.keys():
        if config[key] == 'None':
            config[key] = None

    return config


def read_ld_coefs(filename, wavebin_low, wavebin_up, ld_model='quadratic'):
    """Unpack limb darkening coefficients and interpolate to the wavelength
    grid of data being fit.

    Parameters
    ----------
    filename : str
        Path to file containing model limb darkening coefficients.
    wavebin_low : array-like[float]
        Lower edge of wavelength bins being fit.
    wavebin_up : array-like[float]
        Upper edge of wavelength bins being fit.
    ld_model : str
        Limb darkening model.

    Returns
    -------
    prior_q1 : array-like[float]
        Model estimates for q1 parameter.
    prior_q2 : array-like[float]
        Model estimates for q2 parameter.
    """

    # Open the LD model file and convert c1 and c2 parameters to q1 and q2 of
    # the Kipping (2013) parameterization.
    ld = pd.read_csv(filename, comment='#', sep=',')
    q1s, q2s = juliet.reverse_q_coeffs(ld_model, ld['c1'].values,
                                       ld['c2'].values)

    # Get model wavelengths and sort in increasing order.
    waves = ld['wave'].values
    ii = np.argsort(waves)
    waves = waves[ii]
    q1s, q2s = q1s[ii], q2s[ii]

    prior_q1, prior_q2 = [], []
    # Loop over all fitting bins. Calculate mean of model LD coefs within that
    # range.
    for wl, wu in zip(wavebin_low, wavebin_up):
        current_q1, current_q2 = [], []
        for w, q1, q2 in zip(waves, q1s, q2s):
            if wl < w <= wu:
                current_q1.append(q1)
                current_q2.append(q2)
            elif w > wu:
                prior_q1.append(np.nanmean(current_q1))
                prior_q2.append(np.nanmean(current_q2))
                break

    return prior_q1, prior_q2


def remove_nans(datafile):
    """Remove any NaN values remaining in a datamodel, either in the flux or
    flux error arrays, before passing to an ATOCA extraction.

    Parameters
    ----------
    datafile : str, RampModel
        Datamodel, or path to the datamodel.

    Returns
    -------
    modelout : RampModel
        Input datamodel with NaN values replaced.
    """

    datamodel = open_filetype(datafile)
    modelout = datamodel.copy()
    # Find pixels where either the flux or error is NaN-valued.
    ind = (~np.isfinite(datamodel.data)) | (~np.isfinite(datamodel.err))
    # Set the flux to zero.
    modelout.data[ind] = 0
    # Set the error to an arbitrarily high value.
    modelout.err[ind] = np.nanmedian(datamodel.err) * 10
    # Mark the DQ array to not use these pixels.
    modelout.dq[ind] += 1

    return modelout


def sigma_clip_lightcurves(flux, ferr, thresh=5):
    """Sigma clip outliers in wavelength from final lightcurves.

    Parameters
    ----------
    flux : array-like[float]
        Flux array.
    ferr : array-like[float]
        Flux error array.
    thresh : int
        Sigma level to be clipped.

    Returns
    -------
    flux_clipped : array-like[float]
        Flux array with outliers
    """

    flux_clipped = np.copy(flux)
    nints, nwave = np.shape(flux)
    clipsum = 0
    # Loop over all integrations, and set pixels which deviate by more than
    # the given threshold from the median lightcurve by the median value.
    for itg in range(nints):
        med = np.nanmedian(flux[itg])
        ii = np.where(np.abs(flux[itg] - med) / ferr[itg] > thresh)[0]
        flux_clipped[itg, ii] = med
        clipsum += len(ii)

    print('{0} pixels clipped ({1:.3f}%)'.format(clipsum, clipsum / nints / nwave * 100))

    return flux_clipped


def soss_stability(cube, nsteps=501, axis='x', smoothing_scale=None):
    """ Perform a CCF analysis to track the movement of the SOSS trace
    relative to the median stack over the course of a TSO.

    Parameters
    ----------
    cube : array-like[float]
        Data cube. Should be 3D (ints, dimy, dimx).
    nsteps : int
        Number of CCF steps to test.
    axis : str
        Axis over which to calculate the CCF - either 'x', or 'y'.
    smoothing_scale : int
        Length scale over which to smooth results.

    Returns
    -------
    ccf : array-like[float]
        The cross-correlation results.
    """

    # Get data dimensions.
    nints, dimy, dimx = np.shape(cube)

    # Subtract integration-wise median from cube for CCF.
    cube_sub = cube - np.nanmedian(cube, axis=(1, 2))[:, None, None]
    # Calculate median stack.
    med = bn.nanmedian(cube_sub, axis=0)

    # Initialize CCF variables.
    ccf = np.zeros((nints, nsteps))
    f = interp2d(np.arange(dimx), np.arange(dimy), med, kind='cubic')
    # Perform cross-correlation over desired axis.
    for i in tqdm(range(nints)):
        for j, jj in enumerate(np.linspace(-0.01, 0.01, nsteps)):
            if axis == 'x':
                interp = f(np.arange(dimx) + jj, np.arange(dimy))
            elif axis == 'y':
                interp = f(np.arange(dimx), np.arange(dimy) + jj)
            else:
                msg = 'Unknown axis: {}'.format(axis)
                raise ValueError(msg)
            ccf[i, j] = np.nansum(cube_sub[i] * interp)

    # Determine the peak of the CCF for each integration to get the
    # best-fitting shift.
    maxvals = []
    for i in range(nints):
        maxvals.append(np.where(ccf[i] == np.max(ccf[i]))[0])
    maxvals = np.array(maxvals)
    # Smooth results.
    if smoothing_scale is None:
        smoothing_scale = int(0.2 * nints)
    ccf = median_filter(np.linspace(-0.01, 0.01, nsteps)[maxvals],
                        smoothing_scale)
    ccf = ccf.reshape(nints)

    return ccf


def soss_stability_fwhm(cube, ycens_o1):
    """Estimate the FWHM of the trace over the course of a TSO by fitting a
    Gaussian to each detector column.

    Parameters
    ----------
    cube : array-like[float]
        Data cube. Should be 3D (ints, dimy, dimx).
    ycens_o1 : arrray-like[float]
        Y-centroid positions of the order 1 trace. Should have length dimx.

    Returns
    -------
    fwhm : array-like[float]
        FWHM estimates for each column at every integration.
    """

    def gauss(x, *p):
        amp, mu, sigma = p
        return amp * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))

    # Get data dimensions.
    nints, dimy, dimx = np.shape(cube)
    # Initialize storage array for widths.
    fwhm = np.zeros((nints, dimx-254))

    # Fit a Gaussian to the PSF in each detector column.
    for j in tqdm(range(nints)):
        # Cut out first 500 columns as there is order 2 contmination.
        for i in range(250, dimx-4):
            p0 = [1., ycens_o1[i], 1.]
            data = np.copy(cube[j, :, i])
            # Replace any NaN values with a median.
            if np.isnan(data).any():
                ii = np.where(np.isnan(data))
                data[ii] = np.nanmedian(data)
            # Fit a Gaussian to the profile, and save the FWHM.
            try:
                coeff, var_matrix = curve_fit(gauss, np.arange(dimy), data,
                                              p0=p0)
                fwhm[j, i-250] = coeff[2] * 2.355
            except RuntimeError:
                fwhm[j, i-250] = np.nan

    # Get median FWHM per integration.
    fwhm = np.nanmedian(fwhm, axis=1)
    fwhm -= np.median(fwhm)
    # Smooth the trend.
    fwhm = median_filter(fwhm, int(0.2 * nints))

    return fwhm


def unpack_input_directory(indir, filetag='', exposure_type='CLEAR'):
    """Get all segment files of a specified exposure type from an input data
     directory.

    Parameters
    ----------
    indir : str
        Path to input directory.
    filetag : str
        File name extension of files to unpack.
    exposure_type : str
        Either 'CLEAR' or 'F277W'; unpacks the corresponding exposure type.

    Returns
    -------
    segments: array-like[str]
        File names of the requested exposure and file tag in chronological
        order.
    """

    if indir[-1] != '/':
        indir += '/'
    all_files = glob.glob(indir + '*')
    segments = []

    # Check all files in the input directory to see if they match the
    # specified exposure type and file tag.
    for file in all_files:
        try:
            header = fits.getheader(file, 0)
        # Skip directories or non-fits files.
        except(OSError, IsADirectoryError):
            continue
        # Keep files of the correct exposure with the correct tag.
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


def unpack_spectra(datafile, quantities=('WAVELENGTH', 'FLUX', 'FLUX_ERROR')):
    """Unpack useful quantities from extract1d outputs.

    Parameters
    ----------
    datafile : str, MultiSpecModel
        Extract1d output, or path to the file.
    quantities : tuple(str)
        Quantities to unpack.

    Returns
    -------
    all_spec : dict
        Dictionary containing unpacked quantities for each order.
    """

    multi_spec = open_filetype(datafile)

    # Initialize output dictionary.
    all_spec = {sp_ord: {quantity: [] for quantity in quantities}
                for sp_ord in [1, 2, 3]}
    # Unpack desired quantities into dictionary.
    for spec in multi_spec.spec:
        sp_ord = spec.spectral_order
        for quantity in quantities:
            all_spec[sp_ord][quantity].append(spec.spec_table[quantity])
    for sp_ord in all_spec:
        for key in all_spec[sp_ord]:
            all_spec[sp_ord][key] = np.array(all_spec[sp_ord][key])

    multi_spec.close()

    return all_spec


def verify_path(path):
    """Verify that a given directory exists. If not, create it.

    Parameters
    ----------
    path : str
        Path to directory.
    """

    if os.path.exists(path):
        pass
    else:
        # If directory doesn't exist, create it.
        os.mkdir(path)
