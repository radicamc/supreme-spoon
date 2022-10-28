#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Jul 21 17:33 2022

@author: MCR

Custom JWST DMS pipeline steps for Stage 3 (1D spectral extraction).
"""

from astropy.io import fits
import glob
import numpy as np
import warnings

from applesoss import applesoss

from jwst import datamodels
from jwst.extract_1d.soss_extract import soss_solver
from jwst.pipeline import calwebb_spec2

from supreme_spoon import utils


class SpecProfileStep:
    """Wrapper around custom SpecProfile Reference Construction step.
    """

    def __init__(self, datafiles, output_dir='./'):
        """Step initializer.
        """

        self.output_dir = output_dir
        self.datafiles = np.atleast_1d(datafiles)
        # Get subarray identifier.
        temp_data = datamodels.open(datafiles[0])
        if np.shape(temp_data.data)[0] == 96:
            self.subarray = 'SUBSTRIP96'
        else:
            self.subarray = 'SUBSTRIP256'
        temp_data.close()

    def run(self, save_results=True, force_redo=False):
        """Method to run the step.
        """

        all_files = glob.glob(self.output_dir + '*')
        # If an output file for this segment already exists, skip the step.
        expected_file = self.output_dir + 'APPLESOSS_ref_2D_profile_{}_os1_pad0.fits'.format(self.subarray)
        if expected_file in all_files and force_redo is False:
            print('Output file {} already exists.'.format(expected_file))
            print('Skipping SpecProfile Reference Construction Step.\n')
            specprofile = datamodels.open(expected_file)
            filename = expected_file
        # If no output files are detected, run the step.
        else:
            step_results = specprofilestep(self.datafiles,
                                           save_results=save_results,
                                           output_dir=self.output_dir)
            specprofile, filename = step_results
            filename = self.output_dir + filename

        return specprofile, filename


class SossSolverStep:
    """Wrapper around custom SOSS Solver step.
    """

    def __init__(self, datafiles, deepframe):
        """Step initializer.
        """

        self.datafiles = np.atleast_1d(datafiles)
        self.deepframe = deepframe

    def run(self):
        """Method to run the step.
        """

        transform = sosssolverstep(self.datafiles[0], self.deepframe)

        return transform


class Extract1DStep:
    """Wrapper around default calwebb_spec2 1D Spectral Extraction step, with
    custom modifications.
    """

    def __init__(self, input_data, extract_method, deepframe, smoothed_wlc,
                 output_dir='./'):
        """Step initializer.
        """

        self.tag = 'extract1dstep_{}.fits'.format(extract_method)
        self.output_dir = output_dir
        self.datafiles = np.atleast_1d(input_data)
        self.fileroots = utils.get_filename_root(self.datafiles)
        self.extract_method = extract_method
        self.scaled_deep = deepframe[None, :, :] * smoothed_wlc[:, None, None]

    def run(self, soss_transform, soss_width=25, specprofile=None,
            soss_estimate=None, save_results=True, force_redo=False,
            soss_tikfac=None):
        """Method to run the step.
        """

        print('\nStarting 1D extraction using the {} method.\n'.format(self.extract_method))

        # Initialize loop and storange variables.
        i = 0
        redo = False
        results = []
        completed_segments, redo_segments = [], []
        all_files = glob.glob(self.output_dir + '*')

        # Calculate time axis. For some reason the extrat1d outputs lose the
        # timestamps, so this must be done before extracting.
        times = utils.get_timestamps(self.datafiles)

        # To accomodate the need to occasionally iteratively run the ATOCA
        # extraction, extract segments as long as all segments are not
        # extracted. This is irrelevant for box extractions.
        while len(completed_segments) < len(self.datafiles):
            # If the segment counter gets larger than the number of
            # segments, reset it.
            if i == len(self.datafiles):
                i = i % len(self.datafiles)
                redo = True
            # If segment counter has been reset, but segment has already been
            # successfully extracted, skip it.
            if redo is True and i not in redo_segments:
                i += 1
                continue

            segment = utils.open_filetype(self.datafiles[i])
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                print('Output file {} already exists.'.format(expected_file))
                print('Skipping 1D Extraction Step.\n')
                res = datamodels.open(expected_file)

                if self.extract_method == 'atoca' and soss_estimate is None:
                    atoca_spectra = self.output_dir + self.fileroots[i] + 'AtocaSpectra.fits'
                    soss_estimate = utils.get_soss_estimate(atoca_spectra,
                                                            output_dir=self.output_dir)

            # If no output file is detected, run the step.
            else:
                # Initialize extraction parameters for ATOCA.
                if self.extract_method == 'atoca':
                    soss_atoca = True
                    soss_modelname = self.fileroots[i][:-1]
                    soss_bad_pix = 'model'
                    segment = utils.remove_nans(segment)
                elif self.extract_method == 'box':
                    # Initialize extraction parameters for box.
                    soss_atoca = False
                    soss_modelname = None
                    soss_bad_pix = 'masking'
                    # Replace all remaining bad pixels using scaled median,
                    # and set dq values to zero.
                    istart = segment.meta.exposure.integration_start - 1
                    iend = segment.meta.exposure.integration_end
                    for ii, itg in enumerate(range(istart, iend)):
                        to_replace = np.where(segment.dq[ii] != 0)
                        segment.data[ii][to_replace] = self.scaled_deep[itg][to_replace]
                        segment.dq[ii][to_replace] = 0
                else:
                    msg = ('Invalid extraction: {}'.format(self.extract_method))
                    raise ValueError(msg)

                # Perform the extraction.
                step = calwebb_spec2.extract_1d_step.Extract1dStep()
                try:
                    res = step.call(segment, output_dir=self.output_dir,
                                    save_results=save_results,
                                    soss_transform=[soss_transform[0],
                                                    soss_transform[1],
                                                    soss_transform[2]],
                                    soss_atoca=soss_atoca,
                                    subtract_background=False,
                                    soss_bad_pix=soss_bad_pix,
                                    soss_width=soss_width,
                                    soss_modelname=soss_modelname,
                                    override_specprofile=specprofile,
                                    soss_tikfac=soss_tikfac)
                    # If the step ran successfully, and ATOCA was used, save
                    # the AtocaSpectra output for potential use as the
                    # soss_estimate for later segments.
                    if self.extract_method == 'atoca' and soss_estimate is None:
                        atoca_spectra = self.output_dir + self.fileroots[i] + 'AtocaSpectra.fits'
                        soss_estimate = utils.get_soss_estimate(atoca_spectra,
                                                                output_dir=self.output_dir)
                # When using ATOCA, sometimes a very specific error pops up
                # when an initial estimate of the stellar spectrum cannot be
                # obtained. This is needed to establish the wavelength grid
                # (which has a varying resolution to better capture sharp
                # features in stellar spectra). In these cases, the SOSS
                # estimate provides information to create a wavelength grid.
                except Exception as err:
                    if str(err) == '(m>k) failed for hidden m: fpcurf0:m=0':
                        # If no soss estimate is available, skip this segment
                        # and move to the next one. We will come back and deal
                        # with it later.
                        if soss_estimate is None:
                            print('Initial flux estimate failed, and no soss '
                                  'estimate provided. Moving to next segment.')
                            redo_segments.append(i)
                            i += 1
                            # If all segments fail without a soss estimate,
                            # just fail.
                            if len(redo_segments) == len(self.datafiles):
                                print('No segment can be correctly processed.')
                                raise err
                            continue
                        # Retry extraction with soss estimate.
                        print('\nInitial flux estimate failed, retrying with '
                              'soss_estimate.\n')
                        res = step.call(segment, output_dir=self.output_dir,
                                        save_results=save_results,
                                        soss_transform=[soss_transform[0],
                                                        soss_transform[1],
                                                        soss_transform[2]],
                                        soss_atoca=soss_atoca,
                                        subtract_background=False,
                                        soss_bad_pix=soss_bad_pix,
                                        soss_width=soss_width,
                                        soss_modelname=soss_modelname,
                                        override_specprofile=specprofile,
                                        soss_estimate=soss_estimate,
                                        soss_tikfac=soss_tikfac)
                    # If any other error pops up, raise it.
                    else:
                        raise err
                # Hack to fix file names
                res = utils.fix_filenames(res, '_badpixstep_', self.output_dir,
                                          to_add=self.extract_method)[0]
            results.append(utils.open_filetype(res))
            # If segment was correctly processed, note the segment.
            completed_segments.append(i)
            i += 1

        # Sort the segments in chronological order, in case they were
        # processed out of order.
        seg_nums = [seg.meta.exposure.segment_number for seg in results]
        ii = np.argsort(seg_nums)
        results = np.array(results)[ii]

        # Save the final extraction parameters.
        extract_params = {'transform_x': soss_transform[0],
                          'transform_y': soss_transform[1],
                          'transform_t': soss_transform[2],
                          'soss_width': soss_width,
                          'method': self.extract_method}

        return results, extract_params, times


class LightCurveStep:
    """Wrapper around custom Light Curve Construction step.
    """

    def __init__(self, datafiles, extract_dict, baseline_ints, times,
                 occultation_type='transit', output_dir='./'):
        """Step initializer.
        """

        self.output_dir = output_dir
        self.datafiles = np.atleast_1d(datafiles)
        self.extract_dict = extract_dict
        self.baseline_ints = baseline_ints
        self.occultation_type = occultation_type
        self.times = times

    def run(self, save_results=True):
        """Method to run the step.
        """

        stellar_spectra = lightcurvestep(self.datafiles, times=self.times,
                                         extract_params=self.extract_dict,
                                         baseline_ints=self.baseline_ints,
                                         occultation_type=self.occultation_type,
                                         save_results=save_results,
                                         output_dir=self.output_dir)

        return stellar_spectra


def lightcurvestep(datafiles, times, baseline_ints, extract_params,
                   output_dir='./', save_results=True,
                   occultation_type='transit'):
    """Upack the outputs of the 1D extraction and format them into lightcurves
    at the native detector resolution.

    Parameters
    ----------
    datafiles : array-like[str], array-like[MultiSpecModel]
        Input extract1d data files.
    times : array-like[float]
        Time stamps corresponding to each integration.
    baseline_ints : array-like[int]
        Integrations of ingress and egress.
    output_dir : str
        Directory to which to save outputs.
    save_results : bool
        If True, save outputs to file.
    extract_params : dict
        Dictonary of parameters used for the 1D extraction.
    occultation_type : str
        Type of occultation, either 'transit' or 'eclipse'.

    Returns
    -------
    stellar_spectra : dict
        1D stellar spectra at the native detector resolution.
    """

    print('Constructing stellar spectra.')
    datafiles = np.atleast_1d(datafiles)
    # Format the baseline frames - either out-of-transit or in-eclipse.
    baseline_ints = utils.format_out_frames(baseline_ints,
                                            occultation_type)
    # Calculate the DN/s to e- conversion factor for this TSO.
    dn2e = utils.get_dn2e(datafiles[0])

    # Open the datafiles, and pack the wavelength, flux, and flux error
    # information into data cubes.
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
    # Create 1D wavelength axes from the 2D wavelength solution.
    wave1d_o1, wave1d_o2 = wave2d_o1[0], wave2d_o2[0]

    # Calculate the baseline flux level, and normalize light curves.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        norm_factor_o1 = np.nanmedian(flux_o1[baseline_ints], axis=0)
        nflux_o1 = flux_o1 / norm_factor_o1
        nferr_o1 = ferr_o1 / norm_factor_o1
        norm_factor_o2 = np.nanmedian(flux_o2[baseline_ints], axis=0)
        nflux_o2 = flux_o2 / norm_factor_o2
        nferr_o2 = ferr_o2 / norm_factor_o2

    # Clip remaining 5-sigma outliers.
    nflux_o1_clip = utils.sigma_clip_lightcurves(nflux_o1, nferr_o1)
    nflux_o2_clip = utils.sigma_clip_lightcurves(nflux_o2, nferr_o2)

    # Return the light curves back to their un-normalized state.
    flux_o1_clip = nflux_o1_clip * norm_factor_o1
    flux_o2_clip = nflux_o2_clip * norm_factor_o2

    # Pack the lightcurves into the output format.
    # First get the target name from the orignal data file headers.
    if isinstance(datafiles[0], str):
        filename = datafiles[0]
    else:
        filename = output_dir + datamodels.open(datafiles[0]).meta.filename
    old_header = fits.getheader(filename, 0)
    target_name = old_header['TARGNAME']
    # Pack 1D extraction parameters into the output file header.
    filename = output_dir + target_name + '_' + extract_params['method'] + '_spectra_fullres.fits'
    header_dict, header_comments = utils.get_default_header()
    header_dict['Target'] = target_name
    header_dict['Contents'] = 'Full resolution stellar spectra'
    header_dict['Method'] = extract_params['method']
    header_dict['Width'] = extract_params['soss_width']
    header_dict['Transx'] = extract_params['transform_x']
    header_dict['Transy'] = extract_params['transform_y']
    header_dict['Transth'] = extract_params['transform_t']
    # Calculate the limits of each wavelength bin.
    nint = np.shape(flux_o1_clip)[0]
    wl1, wu1 = utils.get_wavebin_limits(wave1d_o1)
    wl2, wu2 = utils.get_wavebin_limits(wave1d_o2)
    wl1 = np.repeat(wl1[np.newaxis, :], nint, axis=0)
    wu1 = np.repeat(wu1[np.newaxis, :], nint, axis=0)
    wl2 = np.repeat(wl2[np.newaxis, :], nint, axis=0)
    wu2 = np.repeat(wu2[np.newaxis, :], nint, axis=0)

    # Pack the stellar spectra and save to file if requested.
    stellar_spectra = utils.pack_spectra(filename, wl1, wu1, flux_o1_clip,
                                         ferr_o1, wl2, wu2, flux_o2_clip,
                                         ferr_o2, times, header_dict,
                                         header_comments,
                                         save_results=save_results)

    return stellar_spectra


def sosssolverstep(datafile, deepframe):
    """Determine the rotation, as well as vertical and horizontal offsets
    necessary to match the observed trace to the reference files.

    Parameters
    ----------
    deepframe : array-like[float]
        Median baseline stack.
    datafile : str, jwst.datamodel
        Datamodel, or path to datamodel for one segment.

    Returns
    -------
    transform : tuple
        dx, dy, and dtheta transformation.
    """

    print('Solving the SOSS transform.')
    # Get the spectrace reference file to extract the reference centroids.
    step = calwebb_spec2.extract_1d_step.Extract1dStep()
    spectrace_ref = step.get_reference_file(datafile, 'spectrace')
    spec_trace = datamodels.SpecTraceModel(spectrace_ref)
    # Extract the reference centroids for the first and second order.
    xref_o1 = spec_trace.trace[0].data['X']
    yref_o1 = spec_trace.trace[0].data['Y']
    xref_o2 = spec_trace.trace[1].data['X']
    yref_o2 = spec_trace.trace[1].data['Y']

    # Determine the correct subarray identifier.
    if np.shape(deepframe)[0] == 96:
        subarray = 'SUBSTRIP96'
    else:
        subarray = 'SUBSTRIP256'
    # Determine the necessary dx, dy, dtheta transform.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        transform = soss_solver.solve_transform(deepframe, np.isnan(deepframe),
                                                xref_o1, yref_o1,
                                                xref_o2, yref_o2,
                                                soss_filter=subarray,
                                                is_fitted=(True, True, True),
                                                guess_transform=(0, 0, 0))
    print('Determined a transform of:\nx = {}\ny = {}\ntheta = {}'.format(*transform))

    return transform


def specprofilestep(datafiles, save_results=True, output_dir='./'):
    """Wrapper around the APPLESOSS module to construct a specprofile
    reference file tailored to the particular TSO being analyzed.

    Parameters
    ----------
    datafiles : array-like[str], array-like[jwst.RampModel]
        Input datamodels or paths to datamodels for each segment.
    save_results : bool
        If True, save results to file.
    output_dir : str
        Directory to which to save outputs.

    Returns
    -------
    spat_prof : applesoss.EmpiricalProfile object
        Modelled spatial profiles for all orders.
    filename : str
        Name of the output file.
    """

    print('Starting SpecProfile Construction Step.')
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
    deepstack = utils.make_deepstack(cube)

    # Initialize and run the APPLESOSS module with the median stack.
    spat_prof = applesoss.EmpiricalProfile(deepstack, tracetable=tracetable,
                                           wavemap=wavemap)
    spat_prof.build_empirical_profile(verbose=1, wave_increment=0.1)

    # Save results to file if requested.
    if save_results is True:
        if np.shape(deepstack)[0] == 96:
            subarray = 'SUBSTRIP96'
        else:
            subarray = 'SUBSTRIP256'
        filename = spat_prof.write_specprofile_reference(subarray,
                                                         output_dir=output_dir)
    else:
        filename = None

    return spat_prof, filename


def run_stage3(results, deepframe, baseline_ints, smoothed_wlc,
               save_results=True, root_dir='./', force_redo=False,
               extract_method='box', specprofile=None, soss_estimate=None,
               soss_width=25, output_tag='', use_applesoss=True,
               occultation_type='transit', soss_tikfac=None):
    """Run the supreme-SPOON Stage 3 pipeline: 1D spectral extraction, using
    a combination of official STScI DMS and custom steps.

    Parameters
    ----------
    results : array-like[str], array-like[CubeModel]
        supreme-SPOON Stage 2 outputs for each segment.
    deepframe : array-like[float]
        Median out-of-transit stack.
    baseline_ints : array-like[int]
        Integration number of ingress and egress.
    smoothed_wlc : array-like[float]
        Estimate of the normalized light curve.
    save_results : bool
        If True, save the results of each step to file.
    root_dir : str
        Directory from which all relative paths are defined.
    force_redo : bool
        If True, redo steps even if outputs files are already present.
    extract_method : str
        Either 'box' or 'atoca'. Runs the applicable 1D extraction routine.
    specprofile : str, None
        For ATOCA; specprofile reference file.
    soss_estimate : str, None
        For ATOCA; soss estimate file.
    soss_width : int
        Width around the trace centroids, in pixels, for the 1D extraction.
    output_tag : str
        Name tag to append to pipeline outputs directory.
    use_applesoss : bool
        If True, create a tailored specprofile reference file with the
        applesoss module.
    occultation_type : str
        Type of occultation, either 'transit' or 'eclipse'.
    soss_tikfac : int, None
        Tikhonov regularization factor.

    Returns
    -------
    stellar_specra : dict
        1D stellar spectra for each wavelength bin at the native detector
        resolution.
    """

    # ============== DMS Stage 3 ==============
    # 1D spectral extraction.
    print('\n\n**Starting supreme-SPOON Stage 3**')
    print('1D spectral extraction\n\n')

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
            if use_applesoss is True:
                step = SpecProfileStep(results, output_dir=outdir)
                specprofile = step.run(force_redo=force_redo)[1]
            else:
                msg = 'The default specprofile reference file will be used' \
                      ' for extraction.\n' \
                      'For optimal results, consider using a tailored' \
                      ' specprofile reference'
                print(msg)

    # ===== SOSS Solver Step =====
    # Custom DMS step.
    if specprofile is not None:
        soss_transform = [0, 0, 0]
    else:
        step = SossSolverStep(results, deepframe=deepframe)
        soss_transform = step.run()

    # ===== 1D Extraction Step =====
    # Custom/default DMS step.
    step = Extract1DStep(results, deepframe=deepframe,
                         smoothed_wlc=smoothed_wlc,
                         extract_method=extract_method, output_dir=outdir)
    step_results = step.run(soss_transform=soss_transform,
                            soss_width=soss_width, specprofile=specprofile,
                            soss_estimate=soss_estimate,
                            save_results=save_results, force_redo=force_redo,
                            soss_tikfac=soss_tikfac)
    results, extract_params, times = step_results

    # ===== Light Curve Construction Step =====
    # Custom DMS step.
    step = LightCurveStep(results, extract_dict=extract_params, times=times,
                          baseline_ints=baseline_ints,
                          occultation_type=occultation_type, output_dir=outdir)
    stellar_spectra = step.run(save_results=save_results)

    return stellar_spectra
