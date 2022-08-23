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
from tqdm import tqdm
import warnings

from applesoss import applesoss

from jwst import datamodels
from jwst.extract_1d.soss_extract import soss_solver
from jwst.pipeline import calwebb_spec2

from supreme_spoon import plotting, utils

use_applesoss = True


def construct_lightcurves(datafiles, out_frames, output_dir=None,
                          save_results=True, show_plots=False,
                          extract_params=None):
    """Upack the outputs of the 1D extraction and format them into lightcurves
    at the native detector resolution.

    Parameters
    ----------
    datafiles : list[str], list[MultiSpecModel]
        Input extract1d data files.
    out_frames : list[int]
        Integrations of ingress and egress.
    output_dir : str
        Directory to which to save outputs.
    save_results : bool
        If True, save outputs to file.
    show_plots : bool
        If True, show diagnostic plots.
    extract_params : dict
        Dictonary of parameters used for the 1D extraction.

    Returns
    -------
    stellar_spectra : np.array
        1D stellar spectra at the native detector resolution.
    """

    print('Constructing stellar spectra.')
    datafiles = np.atleast_1d(datafiles)
    # Calculate the DN/s to e- conversion factor for this TSO.
    dn2e = utils.get_dn2e(datafiles[0])

    # Open the datafiles, and pack the wavelength, flux, and flux error
    # information into data cubes.
    for i, file in enumerate(datafiles):
        segment = utils.unpack_spectra(file)
        if isinstance(file, str):
            file = datamodels.open(file)
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

    # Generate the time axis of the TSO.
    if isinstance(file, str):
        t = utils.make_time_axis(file)
    else:
        filename = output_dir + file.meta.filename
        t = utils.make_time_axis(filename)

    # Format out of transit integrations.
    out_frames = np.abs(out_frames)
    out_trans = np.concatenate([np.arange(out_frames[0]),
                                np.arange(out_frames[1]) - out_frames[1]])

    # Calculate the out-of-transit flux level, and normalize light curves.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        norm_factor_o1 = np.nanmedian(flux_o1[out_trans], axis=0)
        nflux_o1 = flux_o1 / norm_factor_o1
        nferr_o1 = ferr_o1 / norm_factor_o1
        norm_factor_o2 = np.nanmedian(flux_o2[out_trans], axis=0)
        nflux_o2 = flux_o2 / norm_factor_o2
        nferr_o2 = ferr_o2 / norm_factor_o2

    # Clip remaining 5-sigma outliers.
    nflux_o1_clip = utils.sigma_clip_lightcurves(nflux_o1, nferr_o1)
    nflux_o2_clip = utils.sigma_clip_lightcurves(nflux_o2, nferr_o2)

    # If requested, do diagnostic plot of the sigma clip.
    if show_plots is True:
        kwargs = {'vmax': 1e-4, 'vmin': -1e-4}
        plotting.plot_2dlightcurves(wave1d_o1, nflux_o1, wave1d_o2, nflux_o2)
        plotting.plot_2dlightcurves(wave1d_o1, nflux_o1 - nflux_o1_clip,
                                    wave1d_o2, nflux_o2 - nflux_o2_clip,
                                    **kwargs)
        plotting.plot_2dlightcurves(wave1d_o1, nflux_o1_clip, wave1d_o2,
                                    nflux_o2_clip)

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
    extract_method = filename.split('.fits')[0].split('_')[-1]
    filename = output_dir + target_name + '_' + extract_method + '_spectra_fullres.fits'
    header_dict, header_comments = utils.get_default_header()
    header_dict['Target'] = target_name
    header_dict['Contents'] = 'Full resolution stellar spectra'
    header_dict['Method'] = extract_method
    header_dict['Width'] = extract_params['soss_width']
    header_dict['Transx'] = extract_params['transform_x']
    header_dict['Transy'] = extract_params['transform_y']
    header_dict['Transth'] = extract_params['transform_t']
    # Calculate the extent of each wavelength bin.
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
                                         ferr_o2, t, header_dict,
                                         header_comments,
                                         save_results=save_results)

    return stellar_spectra


def specprofilestep(deepframe, save_results=True, output_dir='./'):
    """Wrapper around the APPLESOSS module to construct a specprofile
    reference file tailored to the particular TSO being analyzed.

    Parameters
    ----------
    deepframe : np.array
        Median out-of-transit stack.
    save_results : bool
        If True, save results to file.
    output_dir : str
        Directory to which to save outputs.

    Returns
    -------
    spat_prof : applesoss EmpiricalProfile object
        Modelled spatial profiles for all orders.
    filename : str
        Name of the output file.
    """

    print('Starting Spectral Profile Construction Step.')
    # Initialize and run the APPLESOSS module with the median stack.
    spat_prof = applesoss.EmpiricalProfile(deepframe)
    spat_prof.build_empirical_profile(verbose=1, wave_increment=0.1)

    # Save results to file is requested.
    if save_results is True:
        if np.shape(deepframe[0]) == 96:
            subarray = 'SUBSTRIP96'
        else:
            subarray = 'SUBSTRIP256'
        filename = spat_prof.write_specprofile_reference(subarray,
                                                         output_dir=output_dir)
    else:
        filename = None

    return spat_prof, filename


def get_soss_transform(deepframe, datafile, show_plots=False,
                       save_results=True, output_dir=None):
    """Determine the rotation, as well as vertical and horizontal offsets
    necessary to match the observed trace to the reference files.

    Parameters
    ----------
    deepframe : np.array
        Median out-of-transit stack.
    datafile : str, MultiSpecModel
        Extract1dStep output.
    show_plots : bool
        If True, show diagnostic plots.
    save_results : bool
        If True, save results to file.
    output_dir : str
        Directory to which to save outputs.

    Returns
    -------
    transform : str
        dx, dy, and dtheta transformation.
    """

    print('Solving the SOSS transform.')
    # Determine the correct subarray identifier.
    if np.shape(deepframe)[0] == 96:
        subarray = 'SUBSTRIP96'
    else:
        subarray = 'SUBSTRIP256'
    # Get the spectrace reference file to extract the reference centroids.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        step = calwebb_spec2.extract_1d_step.Extract1dStep()
        spectrace_ref = step.get_reference_file(datafile, 'spectrace')
        spec_trace = datamodels.SpecTraceModel(spectrace_ref)
    # Extract the reference centroids for the first and second order.
    xref_o1 = spec_trace.trace[0].data['X']
    yref_o1 = spec_trace.trace[0].data['Y']
    xref_o2 = spec_trace.trace[1].data['X']
    yref_o2 = spec_trace.trace[1].data['Y']

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

    # If results are to be saved, or diagnostic plot shown, get the actual
    # data centroids.
    if show_plots is True or save_results is True:
        # Get file name root.
        if isinstance(datafile, str):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                datafile = datamodels.open(datafile)
        save_filename = datafile.meta.filename.split('_')[0]
        # Get trace centroids from the data.
        cens = utils.get_trace_centroids(deepframe, subarray,
                                         output_dir=output_dir,
                                         save_results=save_results,
                                         save_filename=save_filename)
        if show_plots is True:
            # Unpack the reference centroids.
            xdat_o1, ydat_o1 = cens[0]
            xdat_o2, ydat_o2 = cens[1]
            xdat_o3, ydat_o3 = cens[2]

            # Transform the reference centroids based on the above transform.
            xtrans_o1, ytrans_o1 = soss_solver.transform_coords(*transform,
                                                                xref_o1, yref_o1)
            xtrans_o2, ytrans_o2 = soss_solver.transform_coords(*transform,
                                                                xref_o2, yref_o2)
            # Do diagnostic plot.
            labels = ['Extracted Centroids', 'Reference Centroids',
                      'Transformed Centroids']
            plotting.do_centroid_plot(deepframe, [xdat_o1, xref_o1, xtrans_o1],
                                      [ydat_o1, yref_o1, ytrans_o1],
                                      [xdat_o2, xref_o2, xtrans_o2],
                                      [ydat_o2, yref_o2, ytrans_o2],
                                      [xdat_o3], [ydat_o3], labels=labels)

    return transform


def run_stage3(results, deepframe, out_frames, save_results=True,
               show_plots=False, root_dir='./', force_redo=False,
               extract_method='box', specprofile=None, soss_estimate=None,
               soss_width=25, soss_transform=None, output_tag=''):
    """Run the supreme-SPOON Stage 3 pipeline: 1D spectral extraction, using
    a combination of official STScI DMS and custom steps.

    Parameters
    ----------
    results : list[str], list[CubeModel]
        supreme-SPOON Stage 2 outputs for each segment.
    deepframe : np.array
        Median out-of-transit stack.
    out_frames : list[int]
        Integration number of ingress and egress.
    save_results : bool
        If True, save the results of each step to file.
    show_plots : bool
        If True, show diagnostic plots.
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
    soss_transform : list[float], None
        dx, dy, dtheta transformation to match the reference files to the
        observed data.
    output_tag : str
        Name tag to append to pipeline outputs directory.

    Returns
    -------
    stellar_specra : np.array
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

    # Get subarray identifier.
    if np.shape(deepframe)[0] == 96:
        subarray = 'SUBSTRIP96'
    else:
        subarray = 'SUBSTRIP256'

    all_files = glob.glob(outdir + '*')
    results = np.atleast_1d(results)
    # Get file root
    fileroots = []
    for file in results:
        if isinstance(file, str):
            data = datamodels.open(file)
        else:
            data = file
        filename_split = data.meta.filename.split('_')
        fileroot = ''
        for chunk in filename_split[:-1]:
            fileroot += chunk + '_'
        fileroots.append(fileroot)

    # ===== SpecProfile Construction Step =====
    # Custom DMS step
    if extract_method == 'atoca':
        if specprofile is None:
            expected_file = outdir + 'APPLESOSS_ref_2D_profile_{}_os1_pad0.fits'.format(subarray)
            if expected_file in all_files and force_redo is False:
                print('Output file {} already exists.'.format(expected_file))
                print('Skipping SpecProfile Construction Step.')
                specprofile = expected_file
                soss_transform = [0, 0, 0]
            elif use_applesoss is True:
                specprofile = specprofilestep(deepframe,
                                              save_results=save_results,
                                              output_dir=outdir)[1]
                specprofile = outdir + specprofile
                soss_transform = [0, 0, 0]
            else:
                msg = 'APPLESOSS module is unavailable, the default specprofile reference file will be used for extraction.\n' \
                      'For optimal results, consider using a tailored specprofile reference'
                print(msg)

    # ===== 1D Extraction Step =====
    # Custom/default DMS step.
    if soss_transform is None:
        soss_transform = get_soss_transform(deepframe, results[0],
                                            show_plots=show_plots,
                                            save_results=save_results,
                                            output_dir=outdir)
    step_tag = 'extract1dstep_{}.fits'.format(extract_method)
    new_results = []
    extract_params = {'transform_x': soss_transform[0],
                      'transform_y': soss_transform[1],
                      'transform_t': soss_transform[2],
                      'soss_width': soss_width}
    i = 0
    completed_segments = []
    redo_segments = []
    redo = False
    while len(completed_segments) < len(results):
        if i == len(results):
            i = i % len(results)
            redo = True
        if redo is True and i not in redo_segments:
            i += 1
            continue
        segment = results[i]
        expected_file = outdir + fileroots[i] + step_tag
        if expected_file in all_files and force_redo is False:
            print('Output file {} already exists.'.format(expected_file))
            print('Skipping 1D Extraction Step.')
            res = expected_file
            if extract_method == 'atoca' and soss_estimate is None:
                atoca_spectra = outdir + fileroots[i] + 'AtocaSpectra.fits'
                soss_estimate = utils.get_soss_estimate(atoca_spectra,
                                                        output_dir=outdir)
        else:
            segment = utils.open_filetype(segment)
            if extract_method == 'atoca':
                soss_atoca = True
                soss_modelname = fileroots[i][:-1]
                soss_bad_pix = 'model'
                segment = utils.remove_nans(segment)
            else:
                soss_atoca = False
                soss_modelname = None
                soss_bad_pix = 'masking'
                # Interpolate all remaining bad pixels
                print('Interpolating remaining bad pixels.')
                for itg in tqdm(range(segment.dq.shape[0])):
                    segment.data[itg] = utils.do_replacement(segment.data[itg],
                                                             segment.dq[itg])[0]
                segment.dq = np.zeros_like(segment.dq)
            step = calwebb_spec2.extract_1d_step.Extract1dStep()
            try:
                res = step.call(segment, output_dir=outdir,
                                save_results=save_results,
                                soss_transform=[soss_transform[0],
                                                soss_transform[1],
                                                soss_transform[2]],
                                soss_atoca=soss_atoca,
                                subtract_background=False,
                                soss_bad_pix=soss_bad_pix,
                                soss_width=soss_width,
                                soss_modelname=soss_modelname,
                                override_specprofile=specprofile)
                if extract_method == 'atoca' and soss_estimate is None:
                    atoca_spectra = outdir + fileroots[i] + 'AtocaSpectra.fits'
                    soss_estimate = utils.get_soss_estimate(atoca_spectra,
                                                            output_dir=outdir)
            except Exception as err:
                if str(err) == '(m>k) failed for hidden m: fpcurf0:m=0':
                    if soss_estimate is None:
                        i += 1
                        if len(redo_segments) == len(results):
                            print('No segment can be correctly processed.')
                            raise err
                        print('Initial flux estimate failed, and no soss '
                              'estimate provided. Moving to next segment.')
                        redo_segments.append(i)
                        continue

                    print('\nInitial flux estimate failed, retrying with soss_estimate.\n')
                    res = step.call(segment, output_dir=outdir,
                                    save_results=save_results,
                                    soss_transform=[soss_transform[0],
                                                    soss_transform[1],
                                                    soss_transform[2]],
                                    soss_atoca=soss_atoca,
                                    subtract_background=False,
                                    soss_bad_pix=soss_bad_pix, soss_width=25,
                                    soss_modelname=soss_modelname,
                                    override_specprofile=specprofile,
                                    soss_estimate=soss_estimate)
                else:
                    raise err
            # Hack to fix file names
            res = utils.fix_filenames(res, '_badpixstep_', outdir,
                                      to_add=extract_method)[0]
        new_results.append(utils.open_filetype(res))
        completed_segments.append(i)
        i += 1
    results = new_results
    seg_nums = [seg.meta.exposure.segment_number for seg in results]
    ii = np.argsort(seg_nums)
    results = np.array(results)[ii]

    # ===== Lightcurve Construction Step =====
    # Custom DMS step.
    stellar_spectra = construct_lightcurves(results, out_frames=out_frames,
                                            output_dir=outdir,
                                            save_results=save_results,
                                            show_plots=show_plots,
                                            extract_params=extract_params)

    return stellar_spectra
