#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Jul 21 17:33 2022

@author: MCR

Custom JWST DMS pipeline steps for Stage 3 (1D spectral extraction).
"""

from astropy.convolution import Gaussian1DKernel, convolve
from astropy.io import fits
import glob
import numpy as np
import pandas as pd
import pastasoss
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import os

from applesoss import applesoss

from jwst import datamodels
from jwst.pipeline import calwebb_spec2

from supreme_spoon import utils, plotting
from supreme_spoon.utils import fancyprint


class SpecProfileStep:
    """Wrapper around custom SpecProfile Reference Construction step.
    """

    def __init__(self, datafiles, output_dir='./'):
        """Step initializer.
        """

        self.output_dir = output_dir
        self.datafiles = np.atleast_1d(datafiles)
        # Get subarray identifier.
        if isinstance(self.datafiles[0], str):
            self.subarray = fits.getheader(self.datafiles[0])['SUBARRAY']
        elif isinstance(self.datafiles[0], datamodels.CubeModel):
            self.subarray = self.datafiles[0].meta.subarray.name

    def run(self, force_redo=False, empirical=True):
        """Method to run the step.
        """

        all_files = glob.glob(self.output_dir + '*')
        # If an output file for this segment already exists, skip the step.
        expected_file = self.output_dir + 'APPLESOSS_ref_2D_profile_{}_os1_pad20.fits'.format(self.subarray)
        if expected_file in all_files and force_redo is False:
            fancyprint('File {} already exists.'.format(expected_file))
            fancyprint('Skipping SpecProfile Reference Construction Step.')
            specprofile = expected_file
        # If no output files are detected, run the step.
        else:
            specprofile = specprofilestep(self.datafiles,
                                          output_dir=self.output_dir,
                                          empirical=empirical)
            specprofile = self.output_dir + specprofile

        return specprofile


class Extract1DStep:
    """Wrapper around default calwebb_spec2 1D Spectral Extraction step, with
    custom modifications.
    """

    def __init__(self, input_data, extract_method, st_teff=None, st_logg=None,
                 st_met=None, planet_letter='b', output_dir='./',
                 baseline_ints=None, timeseries_o1=None, timeseries_o2=None,
                 pixel_masks=None, input_data2=None):
        """Step initializer.
        """

        self.tag = 'extract1dstep_{}.fits'.format(extract_method)
        self.output_dir = output_dir
        self.datafiles = utils.sort_datamodels(input_data)
        self.fileroots = utils.get_filename_root(self.datafiles)
        self.fileroot_noseg = utils.get_filename_root_noseg(self.fileroots)
        self.extract_method = extract_method
        with utils.open_filetype(self.datafiles[0]) as datamodel:
            self.target_name = datamodel.meta.target.catalog_name
        self.pl_name = self.target_name + ' ' + planet_letter
        self.stellar_params = [st_teff, st_logg, st_met]
        self.baseline_ints = baseline_ints
        self.timeseries_o1 = timeseries_o1
        self.timeseries_o2 = timeseries_o2
        self.pixel_masks = pixel_masks
        if input_data2 is not None:
            self.datafiles2 = utils.sort_datamodels(input_data2)
        else:
            self.datafiles2 = None

    def run(self, soss_width=40, specprofile=None, centroids=None,
            oof_width=None, window_oof=False, save_results=True,
            force_redo=False, do_plot=False, show_plot=False):
        """Method to run the step.
        """

        fancyprint('Starting 1D extraction using the {} '
                   'method.'.format(self.extract_method))

        # Initialize loop and storange variables.
        all_files = glob.glob(self.output_dir + '*')
        expected_file = self.output_dir + self.target_name + '_' + \
            self.extract_method + '_spectra_fullres.fits'
        # If an output file already exists, skip the step.
        if expected_file in all_files and force_redo is False:
            fancyprint('File {} already exists.'.format(expected_file))
            fancyprint('Skipping Extract 1D Step.')
            spectra = expected_file
        # If no output file is detected, run the step.
        else:
            # Option 1: ATOCA extraction.
            if self.extract_method == 'atoca':
                if specprofile is None:
                    raise ValueError('specprofile reference file must be '
                                     'provided for ATOCA extraction.')
                results = []
                for i, segment in enumerate(self.datafiles):
                    # Initialize extraction parameters for ATOCA.
                    soss_modelname = self.fileroots[i][:-1]
                    if window_oof is True:
                        fancyprint('Window 1/f correction cannot be '
                                   'performed with ATOCA extraction.',
                                   msg_type='WARNING')
                    # Perform the extraction.
                    step = calwebb_spec2.extract_1d_step.Extract1dStep()
                    res = step.call(segment, output_dir=self.output_dir,
                                    save_results=save_results,
                                    soss_transform=[0, 0, 0],
                                    subtract_background=False,
                                    soss_bad_pix='model',
                                    soss_width=soss_width,
                                    soss_modelname=soss_modelname,
                                    override_specprofile=specprofile)

                    # Verify that filename is correct.
                    if save_results is True:
                        current_name = self.output_dir + res.meta.filename
                        if expected_file != current_name:
                            res.close()
                            os.rename(current_name, expected_file)
                            res = datamodels.open(expected_file)
                    results.append(res)
            # Option 2: Simple aperture extraction.
            elif self.extract_method == 'box':
                if centroids is None:
                    # Attempt to locate and open a centroids file.
                    centroids = self.output_dir[:-2] + '2/' + \
                                self.fileroot_noseg + 'centroids.csv'
                    try:
                        centroids = pd.read_csv(centroids, comment='#')
                    except FileNotFoundError:
                        raise ValueError('Centroids must be provided for box '
                                         'extraction')
                results = box_extract_soss(self.datafiles, centroids,
                                           soss_width, window_oof=window_oof,
                                           oof_width=oof_width,
                                           baseline_ints=self.baseline_ints,
                                           timeseries_o1=self.timeseries_o1,
                                           timeseries_o2=self.timeseries_o2,
                                           pixel_masks=self.pixel_masks,
                                           datafiles2=self.datafiles2,
                                           do_plot=do_plot,
                                           show_plot=show_plot,
                                           output_dir=self.output_dir,
                                           save_results=save_results)
                if soss_width == 'optimize':
                    # Get optimized width.
                    soss_width = int(results[-1])
                results = results[:-1]
            else:
                raise ValueError('Invalid extraction method')

            # Do step plot if requested - only for atoca.
            if do_plot is True and self.extract_method == 'atoca':
                if save_results is True:
                    plot_file = self.output_dir + self.tag.replace('.fits',
                                                                   '.pdf')
                else:
                    plot_file = None
                models = []
                for name in self.fileroots:
                    models.append(self.output_dir + name + 'SossExtractModel.fits')
                plotting.make_decontamination_plot(self.datafiles, models,
                                                   outfile=plot_file,
                                                   show_plot=show_plot)

            # Save the final extraction parameters.
            extract_params = {'soss_width': soss_width,
                              'method': self.extract_method}
            # Get timestamps and pupil wheel position.
            for i, datafile in enumerate(self.datafiles):
                with utils.open_filetype(datafile) as file:
                    this_time = file.int_times['int_mid_BJD_TDB']
                if i == 0:
                    times = this_time
                    pwcpos = file.meta.instrument.pupil_position
                else:
                    times = np.concatenate([times, this_time])

            # Get throughput data.
            step = calwebb_spec2.extract_1d_step.Extract1dStep()
            thpt = step.get_reference_file(self.datafiles[0], 'spectrace')

            # Clip outliers and format extracted spectra.
            st_teff, st_logg, st_met = self.stellar_params
            spectra = format_extracted_spectra(results, times, extract_params,
                                               self.pl_name, st_teff, st_logg,
                                               st_met, throughput=thpt,
                                               pwcpos=pwcpos,
                                               output_dir=self.output_dir,
                                               save_results=save_results)

        return spectra


def specprofilestep(datafiles, empirical=True, output_dir='./'):
    """Wrapper around the APPLESOSS module to construct a specprofile
    reference file tailored to the particular TSO being analyzed.

    Parameters
    ----------
    datafiles : array-like[str], array-like[jwst.RampModel]
        Input datamodels or paths to datamodels for each segment.
    empirical : bool
        If True, construct profiles using only the data. If False, fall back
        on WebbPSF for the trace wings. Note: The current WebbPSF wings are
        known to not accurately match observations. This mode is therefore not
        advised.
    output_dir : str
        Directory to which to save outputs.

    Returns
    -------
    filename : str
        Name of the output file.
    """

    fancyprint('Starting SpecProfile Construction Step.')
    datafiles = np.atleast_1d(datafiles)

    # Get the most up to date trace table file.
    step = calwebb_spec2.extract_1d_step.Extract1dStep()
    tracetable = step.get_reference_file(datafiles[0], 'spectrace')
    # Get the most up to date 2D wavemap file.
    step = calwebb_spec2.extract_1d_step.Extract1dStep()
    wavemap = step.get_reference_file(datafiles[0], 'wavemap')

    # Create a new deepstack but using all integrations, not just the baseline.
    for i, file in enumerate(datafiles):
        data = datamodels.open(file)
        if i == 0:
            cube = data.data
        else:
            cube = np.concatenate([cube, data.data])
        data.close()
    deepstack = utils.make_deepstack(cube)

    # Initialize and run the APPLESOSS module with the median stack.
    spat_prof = applesoss.EmpiricalProfile(deepstack, tracetable=tracetable,
                                           wavemap=wavemap, pad=20)
    if empirical is False:
        # Get the date of the observations to use the calculated WFE models
        # from that time.
        obs_date = fits.getheader(datafiles[0])['DATE-OBS']
        spat_prof.build_empirical_profile(verbose=0, empirical=False,
                                          wave_increment=0.1,
                                          obs_date=obs_date)
    else:
        spat_prof.build_empirical_profile(verbose=0)

    # Save results to file (non-optional).
    if np.shape(deepstack)[0] == 96:
        subarray = 'SUBSTRIP96'
    else:
        subarray = 'SUBSTRIP256'
    filename = spat_prof.write_specprofile_reference(subarray,
                                                     output_dir=output_dir)

    return filename


def box_extract_soss(datafiles, centroids, soss_width, oof_width=None,
                     pixel_masks=None, timeseries_o1=None, timeseries_o2=None,
                     baseline_ints=None, window_oof=False, datafiles2=None,
                     do_plot=False, show_plot=False, save_results=True,
                     output_dir='./'):
    """Perform a simple box aperture extraction on SOSS orders 1 and 2.

    Parameters
    ----------
    datafiles : array-like[str], array-like[jwst.RampModel]
        Input datamodels or paths to datamodels for each segment.
    centroids : dict
        Dictionary of centroid positions for all SOSS orders.
    soss_width : int, str
        Width of extraction box. Or 'optimize'.
    oof_width : int
        Width of window to use to calculate 1/f.
    pixel_masks : array-like[str], None
        List of paths to maps of pixels to mask for each data segment. Can be
        3D (nints, dimy, dimx), or 2D (dimy, dimx).
    timeseries_o1 : array-like[float], str, None
        Estimate of normalized light curve(s) for order 1, or path to file.
    timeseries_o2 : array-like[float], str, None
        Estimate of normalized light curve(s) for order 2, or path to file.
    baseline_ints : array-like[int]
        Integration numbers of ingress and egress.
    window_oof : bool
        If True, correct 1/f noise using window around trace before extracting.
    datafiles2 : array-like[str], array-like[jwst.RampModel], None
        Input datamodels for order 2 if chromatic 1/f corrections were applied
        and both orders are to be extracted simultaneously.
    do_plot : bool
        If True, do the step diagnostic plot.
    show_plot : bool
        If True, show the step diagnostic plot instead of/in addition to
        saving it to file.
    output_dir : str
        Directory to which to output results.
    save_results : bool
        If True, save results to file.

    Returns
    -------
    wave_o1 : array_like[float]
        2D wavelength solution for order 1.
    flux_o1 : array_like[float]
        2D extracted flux for order 1.
    ferr_o1: array_like[float]
        2D flux errors for order 1.
    wave_o2 : array_like[float]
        2D wavelength solution for order 2.
    flux_o2 : array_like[float]
        2D extracted flux for order 2.
    ferr_o2 : array_like[float]
        2D flux errors for order 2.
    soss_width : int
        Optimized aperture width.
    """

    datafiles = np.atleast_1d(datafiles)
    # Get flux and errors to extract.
    for i, file in enumerate(datafiles):
        with utils.open_filetype(file) as datamodel:
            if i == 0:
                cube = datamodel.data
                ecube = datamodel.err
            else:
                cube = np.concatenate([cube, datamodel.data])
                ecube = np.concatenate([ecube, datamodel.err])
    # If order 2 specific datafiles are passed, read them in to include them
    # in the extraction.
    if datafiles2 is not None:
        # Unless window 1/f option is selected.
        if window_oof is True:
            fancyprint('Window 1/f correction selected, ignorning passed '
                       'order 2 datafiles.', msg_type='WARNING')
        else:
            for i, file in enumerate(datafiles2):
                with utils.open_filetype(file) as datamodel:
                    if i == 0:
                        cube2 = datamodel.data
                        ecube2 = datamodel.err
                    else:
                        cube2 = np.concatenate([cube2, datamodel.data])
                        ecube2 = np.concatenate([ecube2, datamodel.err])

    # Get centroid positions.
    x1 = centroids['xpos'].values
    y1, y2 = centroids['ypos o1'].values, centroids['ypos o2'].values
    ii = np.where(np.isfinite(y2))
    x2, y2 = x1[ii], y2[ii]

    # ===== Optimize Aperture Width =====
    if soss_width == 'optimize':
        fancyprint('Optimizing extraction width...')
        # Extract order 1 with a variety of widths and find the one that
        # minimizes the white light curve scatter.
        scatter = []
        for w in tqdm(range(10, 61)):
            flux = do_box_extraction(cube, ecube, y1, width=w,
                                     progress=False)[0]
            wlc = np.nansum(flux, axis=1)
            s = np.median(np.abs(0.5*(wlc[0:-2] + wlc[2:]) - wlc[1:-1]))
            scatter.append(s)
        scatter = np.array(scatter)
        # Find the width that minimizes the scatter.
        ii = np.argmin(scatter)
        soss_width = np.linspace(10, 60, 51)[ii]
        fancyprint('Using width of {} pxiels.'.format(int(soss_width)))

        # Do diagnostic plot if requested.
        if do_plot is True:
            if save_results is True:
                outfile = output_dir + '_aperture_optimization.pdf'
            else:
                outfile = None
            plotting.make_soss_width_plot(scatter, ii, outfile=outfile,
                                          show_plot=show_plot)

    # ===== Window 1/f Correction ======
    # Do window 1/f correction.
    if window_oof is True:
        fancyprint('Performing window 1/f correction.')
        # Ensure window is a ~appropriate width.
        if oof_width is None:
            fancyprint('No 1/f window width provided, using default width '
                       'of {}'.format(soss_width+30), msg_type='WARNING')
            oof_width = soss_width + 30
        if oof_width < (soss_width + 10):
            fancyprint('1/f window width must be at least 10 pixels wider '
                       'than extraction width. Using default width '
                       'of {}'.format(soss_width+30), msg_type='WARNING')
            oof_width = soss_width + 30

        # Get timeseries for difference image scaling.
        if timeseries_o1 is None:
            msg = '1D or 2D timeseries must be provided to use window ' \
                  '1/f method.'
            raise ValueError(msg)
        fancyprint('Reading order 1 timeseries file.')
        if isinstance(timeseries_o1, str):
            timeseries_o1 = np.load(timeseries_o1)
        # If passed light curve is 1D, extend to 2D.
        if np.ndim(timeseries_o1) == 1:
            dimx = np.shape(cube)[-1]
            timeseries_o1 = np.repeat(timeseries_o1[:, np.newaxis], dimx,
                                      axis=1)
        # Check if separate timeseries is passed for order 2.
        if timeseries_o2 is None:
            fancyprint('Using the same timeseries for orders 1 and 2.',
                       msg_type='WARNING')
            timeseries_o2 = np.copy(timeseries_o1)
        else:
            fancyprint('Reading order 2 timeseries file.')
            if isinstance(timeseries_o2, str):
                timeseries_o2 = np.load(timeseries_o2)
            # If passed light curve is 1D, extend to 2D.
            if np.ndim(timeseries_o2) == 1:
                dimx = np.shape(cube)[-1]
                timeseries_o2 = np.repeat(timeseries_o2[:, np.newaxis], dimx,
                                          axis=1)

        # Create difference cube.
        assert baseline_ints is not None
        baseline_ints = utils.format_out_frames(baseline_ints)
        deepstack = utils.make_deepstack(cube[baseline_ints])
        diff_cube1 = cube - deepstack[None, :, :] * timeseries_o1[:, None, :]
        diff_cube2 = cube - deepstack[None, :, :] * timeseries_o2[:, None, :]

        # Perform the window 1/f correction seperately for each order.
        fancyprint('Doing order 1 1/f window correction.')
        cube1 = window_oneoverf(cube, diff_cube1, x1, y1, soss_width,
                                oof_width, pixel_masks)
        fancyprint('Doing order 2 1/f window correction.')
        cube2 = window_oneoverf(cube, diff_cube2, x2, y2, soss_width,
                                oof_width, pixel_masks)
        ecube2 = ecube
    elif datafiles2 is not None:
        cube1 = cube
    else:
        cube1 = cube
        cube2 = cube
        ecube2 = ecube

    # ===== Extraction ======
    # Do the extraction.
    fancyprint('Performing simple aperture extraction.')
    fancyprint('Extracting Order 1')
    flux_o1, ferr_o1 = do_box_extraction(cube1, ecube, y1, width=soss_width)
    fancyprint('Extracting Order 2')
    flux_o2, ferr_o2 = do_box_extraction(cube2, ecube2, y2, width=soss_width,
                                         extract_end=len(y2))

    # Get default wavelength solution.
    step = calwebb_spec2.extract_1d_step.Extract1dStep()
    wavemap = step.get_reference_file(datafiles[0], 'wavemap')
    # Remove 20 pixel padding that is there for some reason.
    wave_o1 = np.mean(fits.getdata(wavemap, 1)[20:-20, 20:-20], axis=0)
    wave_o2 = np.mean(fits.getdata(wavemap, 2)[20:-20, 20:-20], axis=0)

    return wave_o1, flux_o1, ferr_o1, wave_o2, flux_o2, ferr_o2, soss_width


def do_box_extraction(cube, err, ypos, width, extract_start=0,
                      extract_end=None, progress=True):
    """Do intrapixel aperture extraction.

    Parameters
    ----------
    cube : array-like(float)
        Data cube.
    err : array-like(float)
        Error cube.
    ypos : array-like(float)
        Detector Y-positions to extract.
    width : int
        Full-width of the extraction aperture to use.
    extract_start : int
        Detector X-position at which to start extraction.
    extract_end : int, None
        Detector X-position at which to end extraction.
    progress : bool
        if True, show extraction progress bar.

    Returns
    -------
    f : np.array(float)
        Extracted flux values.
    ferr : np.array(float)
         Extracted error values.
    """

    # Ensure data and errors are the same shape.
    assert np.shape(cube) == np.shape(err)
    nint, dimy, dimx = np.shape(cube)

    # If extraction end is not specified, extract the whole frame.
    if extract_end is None:
        extract_end = dimx

    # Initialize output arrays.
    f, ferr = np.zeros((nint, dimx)), np.zeros((nint, dimx))

    # Determine the upper and lower edges of the extraction region. Cut at
    # detector edges if necessary.
    edge_up = np.min([ypos + width / 2, np.ones_like(ypos) * dimy], axis=0)
    edge_low = np.max([ypos - width / 2, np.zeros_like(ypos)], axis=0)

    # Loop over all integrations and sum flux within the extraction aperture.
    for i in tqdm(range(nint), disable=not progress):
        for x in range(extract_start, extract_end):
            # First sum the whole pixel components within the aperture.
            up_whole = np.floor(edge_up[x - extract_start]).astype(int)
            low_whole = np.ceil(edge_low[x - extract_start]).astype(int)
            this_flux = np.sum(cube[i, low_whole:up_whole, x])
            this_err = np.sum(err[i, low_whole:up_whole, x]**2)

            # Now incorporate the partial pixels at the upper and lower edges.
            if edge_up[x - extract_start] >= (dimy-1) or edge_low[x - extract_start] == 0:
                continue
            else:
                up_part = edge_up[x - extract_start] % 1
                low_part = 1 - edge_low[x - extract_start] % 1
                this_flux += (up_part * cube[i, up_whole, x] +
                              low_part * cube[i, low_whole, x])
                this_err += (up_part * err[i, up_whole, x]**2 +
                             low_part * err[i, low_whole, x]**2)
                f[i, x] = this_flux
                ferr[i, x] = np.sqrt(this_err)

    return f, ferr


def do_ccf(wave, flux, err, mod_flux, nsteps=1000):
    """Perform a cross-correlation analysis between an extracted and model
    stellar spectrum to determine the appropriate wavelength shift between
    the two.

    Parameters
    ----------
    wave : array-like[float]
        Wavelength axis.
    flux : array-like[float]
        Extracted spectrum.
    err : array-like[float]
        Errors on extracted spectrum.
    mod_flux : array-like[float]
        Model spectrum.
    nsteps : int
        Number of wavelength steps to test.

    Returns
    -------
    shift : float
        Wavelength shift between the model and extracted spectrum in microns.
    """

    def highpass_filter(signal, order=3, freq=0.05):
        """High pass filter."""
        b, a = butter(order, freq, btype='high')
        signal_filt = filtfilt(b, a, signal)
        return signal_filt

    ccf = np.zeros(nsteps)
    # Trim edges off of input data to avoid interplation errors.
    wav_a, flux_a, ferr_a = wave[50:-50], flux[50:-50], err[50:-50]
    # Max-value normalize the model spectrum and initialize interpolation.
    mod_norm = mod_flux / np.max(mod_flux)
    f = interp1d(wave, mod_norm, kind='cubic')
    # Max-value normalize and high-pass filter the data.
    data = highpass_filter(flux_a / np.max(flux_a))
    error = ferr_a / np.max(flux_a)

    # Perform the CCF.
    for j, jj in enumerate(np.linspace(-0.01, 0.01, nsteps)):
        # Calculate new wavelength axis.
        new_wave = wav_a + jj
        # Interpolate model onto new axis and high-pass filter.
        model_interp = f(new_wave)
        model_interp = highpass_filter(model_interp)
        # Calculate the CCF at this step.
        ccf[j] = np.nansum(model_interp * data / error ** 2)

    # Determine the peak of the CCF for each integration to get the
    # best-fitting shift.
    maxval = np.argmax(ccf)
    shift = np.linspace(-0.01, 0.01, nsteps)[maxval]

    return shift


def format_extracted_spectra(datafiles, times, extract_params, target_name,
                             st_teff=None, st_logg=None, st_met=None,
                             throughput=None, pwcpos=None, output_dir='./',
                             save_results=True):
    """Unpack the outputs of the 1D extraction and format them into
    lightcurves at the native detector resolution.

    Parameters
    ----------
    datafiles : array-like[str], array-like[MultiSpecModel], tuple
        Input extract1d data files.
    times : array-like[float]
        Time stamps corresponding to each integration.
    output_dir : str
        Directory to which to save outputs.
    save_results : bool
        If True, save outputs to file.
    extract_params : dict
        Dictonary of parameters used for the 1D extraction.
    target_name : str
        Name of the target.
    st_teff : float, None
        Stellar effective temperature.
    st_logg : float, None
        Stellar log surface gravity.
    st_met : float, None
        Stellar metallicity as [Fe/H].
    throughput : str
        Path to JWST spectrace reference file.
    pwcpos : float
        Filter wheel position.

    Returns
    -------
    spectra : dict
        1D stellar spectra at the native detector resolution.
    """

    fancyprint('Formatting extracted 1d spectra.')
    # Box extract outputs will just be a tuple of arrays.
    if isinstance(datafiles, tuple):
        flux_o1 = datafiles[1]
        ferr_o1 = datafiles[2]
        flux_o2 = datafiles[4]
        ferr_o2 = datafiles[5]
    # Whereas ATOCA extract outputs are in the atoca extract1dstep format.
    else:
        # Open the datafiles, and pack the wavelength, flux, and flux error
        # information into data cubes.
        datafiles = np.atleast_1d(datafiles)
        for i, file in enumerate(datafiles):
            segment = utils.unpack_atoca_spectra(file)
            if i == 0:
                flux_o1 = segment[1]['FLUX']
                ferr_o1 = segment[1]['FLUX_ERROR']
                flux_o2 = segment[2]['FLUX']
                ferr_o2 = segment[2]['FLUX_ERROR']
            else:
                flux_o1 = np.concatenate([flux_o1, segment[1]['FLUX']])
                ferr_o1 = np.concatenate([ferr_o1, segment[1]['FLUX_ERROR']])
                flux_o2 = np.concatenate([flux_o2, segment[2]['FLUX']])
                ferr_o2 = np.concatenate([ferr_o2, segment[2]['FLUX_ERROR']])

    # Get wavelength solution.
    # First, use PASTASOSS to predict wavelength solution from pupil wheel
    # position.
    wave1d_o1 = pastasoss.get_soss_traces(pwcpos=pwcpos, order='1',
                                          interp=True).wavelength
    soln_o2 = pastasoss.get_soss_traces(pwcpos=pwcpos, order='2',
                                        interp=True)
    xpos_o2, wave1d_o2 = soln_o2.x.astype(int), soln_o2.wavelength
    # Trim extracted quantities to match shapes of pastasoss quantities.
    flux_o1 = flux_o1[:, 4:-4]
    ferr_o1 = ferr_o1[:, 4:-4]
    flux_o2 = flux_o2[:, xpos_o2]
    ferr_o2 = ferr_o2[:, xpos_o2]

    # Now cross-correlate with stellar model.
    # If one or more of the stellar parameters are not provided, use the
    # wavelength solution from pastasoss.
    if None in [st_teff, st_logg, st_met]:
        fancyprint('Stellar parameters not provided. '
                   'Using default wavelength solution.', msg_type='WARNING')
    else:
        fancyprint('Refining the wavelength calibration.')
        # Create a grid of stellar parameters, and download PHOENIX spectra
        # for each grid point.
        thisout = output_dir + 'phoenix_models'
        utils.verify_path(thisout)
        res = utils.download_stellar_spectra(st_teff, st_logg, st_met,
                                             outdir=thisout)
        wave_file, flux_files = res
        # Interpolate model grid to correct stellar parameters.
        # Reverse direction of both arrays since SOSS is extracted red to blue.
        mod_flux = utils.interpolate_stellar_model_grid(flux_files, st_teff,
                                                        st_logg, st_met)
        mod_wave = fits.getdata(wave_file)/1e4

        # Convolve model to lower resolution and interpolate to data
        # wavelengths.
        gauss = Gaussian1DKernel(stddev=500)
        mod_flux = convolve(mod_flux, gauss)
        mod_flux = np.interp(wave1d_o1[::-1], mod_wave, mod_flux)[::-1]
        # Add throuput effects to model.
        thpt = fits.open(throughput)
        twave = thpt[1].data['WAVELENGTH']
        thpt = thpt[1].data['THROUGHPUT']
        thpt = np.interp(wave1d_o1[::-1], twave, thpt)[::-1]
        mod_flux *= thpt

        # Cross-correlate extracted spectrum with model to refine wavelength
        # calibration.
        x1d_flux = np.nansum(flux_o1, axis=0)
        x1d_err = np.sqrt(np.nansum(ferr_o1**2, axis=0))
        wave_shift = do_ccf(wave1d_o1, x1d_flux, x1d_err, mod_flux)
        fancyprint('Found a wavelength shift of {}um'.format(wave_shift))
        wave1d_o1 += wave_shift
        wave1d_o2 += wave_shift

    # Restrict order 2 to only the useful bit (0.6 - 0.85µm).
    ii = np.where((wave1d_o2 >= 0.6) & (wave1d_o2 < 0.85))[0]
    wave1d_o2 = wave1d_o2[ii]
    flux_o2 = flux_o2[:, ii]
    ferr_o2 = ferr_o2[:, ii]

    # Clip remaining 3-sigma outliers.
    flux_o1_clip = utils.sigma_clip_lightcurves(flux_o1, ferr_o1)
    flux_o2_clip = utils.sigma_clip_lightcurves(flux_o2, ferr_o2)

    # Pack the lightcurves into the output format.
    # Put 1D extraction parameters in the output file header.
    filename = output_dir + target_name[:-2] + '_' + extract_params['method'] \
        + '_spectra_fullres.fits'
    header_dict, header_comments = utils.get_default_header()
    header_dict['Target'] = target_name[:-2]
    header_dict['Contents'] = 'Full resolution stellar spectra'
    header_dict['Method'] = extract_params['method']
    header_dict['Width'] = extract_params['soss_width']
    # Calculate the limits of each wavelength bin.
    nint = np.shape(flux_o1_clip)[0]
    wl1, wu1 = utils.get_wavebin_limits(wave1d_o1)
    wl2, wu2 = utils.get_wavebin_limits(wave1d_o2)
    wl1 = np.repeat(wl1[np.newaxis, :], nint, axis=0)
    wu1 = np.repeat(wu1[np.newaxis, :], nint, axis=0)
    wl2 = np.repeat(wl2[np.newaxis, :], nint, axis=0)
    wu2 = np.repeat(wu2[np.newaxis, :], nint, axis=0)

    # Pack the stellar spectra and save to file if requested.
    spectra = utils.save_extracted_spectra(filename, wl1, wu1, flux_o1_clip,
                                           ferr_o1, wl2, wu2, flux_o2_clip,
                                           ferr_o2, times, header_dict,
                                           header_comments,
                                           save_results=save_results)

    return spectra


def window_oneoverf(cube, diff_cube, xpos, ypos, soss_width, oof_width,
                    pixel_masks=None):
    """Subtract 1/f noise from the trace of an individual order using a window
    of pixels around the extraction box.

    Parameters
    ----------
    cube : array-like(float)
        Datacube.
    diff_cube : array-like(float)
        Datacube of difference images.
    xpos : array-like(float)
        Trace x positions.
    ypos : array-like(float)
        Trace y positions.
    soss_width : int
        Extractin width.
    oof_width : int
        Width of window to use to calculate 1/f.
    pixel_masks : array-like[str], None
        List of paths to maps of pixels to mask for each data segment. Can be
        3D (nints, dimy, dimx), or 2D (dimy, dimx).

    Returns
    -------
    cube_corr : np.array(float)
        Datacube corrected for 1/f.
    """

    cube_corr = np.ones_like(cube)
    nint, dimy, dimx = np.shape(cube)

    # Get outlier masks.
    if pixel_masks is None:
        fancyprint('No outlier maps passed, ignoring outliers.')
        outliers = np.zeros((nint, dimy, dimx))
    else:
        for i, file in enumerate(pixel_masks):
            fancyprint('Reading outlier map {}'.format(file))
            if i == 0:
                # Get second extension which will be flags without tracemask.
                outliers = fits.getdata(file, 2)
            else:
                outliers = np.concatenate([outliers, fits.getdata(file, 2)])
    outliers = np.where(outliers == 0, 1, np.nan)
    # Apply the outlier mask.
    diff_cube *= outliers

    for x, y in zip(xpos, ypos):
        # Define extent of the window.
        up_w = np.ceil(np.min([y + oof_width//2, dimy-1])).astype(int)
        low_w = np.floor(np.max([y - oof_width//2, 0])).astype(int)
        up_b = np.ceil(np.min([y + soss_width // 2, dimy-1])).astype(int)
        low_b = np.floor(np.max([y - soss_width // 2, 0])).astype(int)
        # Stack the window rows.
        window = np.concatenate([diff_cube[:, low_w:low_b, x],
                                 diff_cube[:, up_b:up_w, x]], axis=1)
        # Subtract the 1/f calculated in the window from the cube column.
        if np.shape(window)[1] != 0:
            cube_corr[:, :, x] = cube[:, :, x] - np.nanmedian(window, axis=1)[:, None]
        # In case the extraction box is the entire frame, don't subtract
        # anything so that this doesn't fail.
        else:
            cube_corr[:, :, x] = cube[:, :, x]

    return cube_corr


def run_stage3(results, save_results=True, root_dir='./', force_redo=False,
               extract_method='box', specprofile=None, centroids=None,
               soss_width=40, st_teff=None, st_logg=None, st_met=None,
               planet_letter='b', output_tag='', do_plot=False,
               show_plot=False, timeseries_o1=None, timeseries_o2=None,
               pixel_masks=None, window_oof=False, oof_width=None,
               baseline_ints=None, datafiles2=None):
    """Run the supreme-SPOON Stage 3 pipeline: 1D spectral extraction, using
    a combination of the official STScI DMS and custom steps.

    Parameters
    ----------
    results : array-like[str], array-like[CubeModel]
        supreme-SPOON Stage 2 outputs for each segment.
    save_results : bool
        If True, save the results of each step to file.
    root_dir : str
        Directory from which all relative paths are defined.
    force_redo : bool
        If True, redo steps even if outputs files are already present.
    extract_method : str
        Either 'box' or 'atoca'. Runs the applicable 1D extraction routine.
    specprofile : str, None
        Specprofile reference file; only neceessary for ATOCA extractions.
    centroids : dict
        Dictionary of centroid positions for SOSS orders; only necessary for
        box extractions.
    soss_width : int
        Width around the trace centroids, in pixels, for the 1D extraction.
    st_teff : float, None
        Stellar effective temperature.
    st_logg : float, None
        Stellar log surface gravity.
    st_met : float, None
        Stellar metallicity as [Fe/H].
    planet_letter : str
        Letter designation for the planet.
    output_tag : str
        Name tag to append to pipeline outputs directory.
    do_plot : bool
        If True, make step diagnostic plot.
    show_plot : bool
        Only necessary if do_plot is True. Show the diagnostic plots in
        addition to/instead of saving to file.
    window_oof : bool
        If True, subtract 1/f noise using a window around the trace before
        extracting.
    oof_width : int
        Width of window to use to calculate 1/f.
    pixel_masks : array-like[str], None
        List of paths to maps of pixels to mask for each data segment. Can be
        3D (nints, dimy, dimx), or 2D (dimy, dimx).
    timeseries_o1 : array-like[float], str, None
        Estimate of normalized light curve(s) for order 1, or path to file.
    timeseries_o2 : array-like[float], str, None
        Estimate of normalized light curve(s) for order 2, or path to file.
    baseline_ints : array-like[int]
        Integration numbers of ingress and egress.
    datafiles2 : array-like[str], array-like[CubeModel]
        If chromatic 1/f correction was performed; extraction ready order 2
        outputs.

    Returns
    -------
    specra : dict
        1D stellar spectra for each wavelength bin at the native detector
        resolution.
    """

    # ============== DMS Stage 3 ==============
    # 1D spectral extraction.
    fancyprint('**Starting supreme-SPOON Stage 3**')
    fancyprint('1D spectral extraction...')

    if output_tag != '':
        output_tag = '_' + output_tag
    # Create output directories and define output paths.
    utils.verify_path(root_dir + 'pipeline_outputs_directory' + output_tag)
    utils.verify_path(root_dir + 'pipeline_outputs_directory' + output_tag + '/Stage3')
    outdir = root_dir + 'pipeline_outputs_directory' + output_tag + '/Stage3/'

    # ===== SpecProfile Construction Step =====
    # Custom DMS step
    if extract_method == 'atoca':
        if specprofile is None:
            step = SpecProfileStep(results, output_dir=outdir)
            specprofile = step.run(force_redo=force_redo)

    # ===== 1D Extraction Step =====
    # Custom/default DMS step.
    step = Extract1DStep(results, extract_method=extract_method,
                         st_teff=st_teff, st_logg=st_logg, st_met=st_met,
                         planet_letter=planet_letter,  output_dir=outdir,
                         baseline_ints=baseline_ints, pixel_masks=pixel_masks,
                         timeseries_o1=timeseries_o1,
                         timeseries_o2=timeseries_o2, input_data2=datafiles2)
    spectra = step.run(soss_width=soss_width, specprofile=specprofile,
                       centroids=centroids, save_results=save_results,
                       force_redo=force_redo, do_plot=do_plot,
                       show_plot=show_plot, window_oof=window_oof,
                       oof_width=oof_width)

    return spectra
