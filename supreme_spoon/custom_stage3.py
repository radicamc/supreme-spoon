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

from jwst import datamodels
from jwst.extract_1d.soss_extract import soss_solver
from jwst.pipeline import calwebb_spec2

from sys import path
applesoss_path = '/home/radica/GitHub/APPLESOSS/'
path.insert(1, applesoss_path)
try:
    from APPLESOSS import applesoss
    use_applesoss = True
except ModuleNotFoundError:
    msg = 'APPLESOSS module not available. Some capabilities will be limited.'
    warnings.warn(msg)
    use_applesoss = False

from supreme_spoon import plotting, utils


def construct_lightcurves(datafiles, out_frames, output_dir=None,
                          save_results=True, show_plots=False,
                          extract_params=None):
    print('Constructing stellar spectra')
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
    if isinstance(file, str):
        t = utils.make_time_axis(file)
    else:
        filename = file.meta.filename
        t = utils.make_time_axis(filename)
    out_frames = np.abs(out_frames)
    out_trans = np.concatenate([np.arange(out_frames[0]),
                                np.arange(out_frames[1]) - out_frames[1]])

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
        plotting.plot_2dlightcurves(wave1d_o1, nflux_o1, wave1d_o2, nflux_o2)
        plotting.plot_2dlightcurves(wave1d_o1, nflux_o1 - nflux_o1_clip,
                                    wave1d_o2, nflux_o2 - nflux_o2_clip,
                                    **kwargs)
        plotting.plot_2dlightcurves(wave1d_o1, nflux_o1_clip, wave1d_o2,
                                    nflux_o2_clip)

    flux_o1_clip = nflux_o1_clip * norm_factor_o1
    flux_o2_clip = nflux_o2_clip * norm_factor_o2

    # Get useful info from header
    if isinstance(datafiles[0], str):
        filename = datafiles[0]
    else:
        filename = output_dir + datamodels.open(datafiles[0]).meta.filename
    old_header = fits.getheader(filename, 0)
    target_name = old_header['TARGNAME']
    extract_method = filename.split('.fits')[0].split('_')[-1]
    # Save full res stellar spectra
    filename = output_dir + target_name + '_' + extract_method + '_spectra_fullres.fits'
    header_dict, header_comments = utils.get_default_header()
    header_dict['Target'] = target_name
    header_dict['Contents'] = 'Full resolution stellar spectra'
    header_dict['Method'] = extract_method
    header_dict['Width'] = extract_params['soss_width']
    header_dict['Transx'] = extract_params['transform_x']
    header_dict['Transy'] = extract_params['transform_y']
    header_dict['Transth'] = extract_params['transform_t']
    nint = np.shape(flux_o1_clip)[0]
    wl1, wu1 = utils.get_wavebin_limits(wave1d_o1)
    wl2, wu2 = utils.get_wavebin_limits(wave1d_o2)
    wl1 = np.repeat(wl1[np.newaxis, :], nint, axis=0)
    wu1 = np.repeat(wu1[np.newaxis, :], nint, axis=0)
    wl2 = np.repeat(wl2[np.newaxis, :], nint, axis=0)
    wu2 = np.repeat(wu2[np.newaxis, :], nint, axis=0)

    stellar_spectra = utils.pack_spectra(filename, wl1, wu1, flux_o1_clip,
                                         ferr_o1, wl2, wu2, flux_o2_clip,
                                         ferr_o2, t, header_dict,
                                         header_comments,
                                         save_results=save_results)

    return stellar_spectra


def specprofilestep(deepframe, save_results=True, output_dir='./'):

    print('Starting Spectral Profile Construction Step')
    spat_prof = applesoss.EmpiricalProfile(deepframe)
    spat_prof.build_empirical_profile(verbose=1, wave_increment=0.1)

    if save_results is True:
        filename = spat_prof.write_specprofile_reference('SUBSTRIP256',
                                                         output_dir=output_dir)
    else:
        filename = None

    return spat_prof, filename


def get_soss_transform(deepframe, datafile, show_plots=False,
                       save_results=True, output_dir=None):

    print('Solving the SOSS transform')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
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
    print('Determined a transform of:\nx = {}\ny = {}\ntheta = {}'.format(*transform))

    if show_plots is True or save_results is True:
        if isinstance(datafile, str):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                datafile = datamodels.open(datafile)
        save_filename = datafile.meta.filename.split('_')[0]
        cens = utils.get_trace_centroids(deepframe, 'SUBSTRIP256',
                                         output_dir=output_dir,
                                         save_results=save_results,
                                         save_filename=save_filename)

        xdat_o1, ydat_o1 = cens[0]
        xdat_o2, ydat_o2 = cens[1]
        xdat_o3, ydat_o3 = cens[2]

        xtrans_o1, ytrans_o1 = soss_solver.transform_coords(*transform,
                                                            xref_o1, yref_o1)
        xtrans_o2, ytrans_o2 = soss_solver.transform_coords(*transform,
                                                            xref_o2, yref_o2)
        labels = ['Extracted Centroids', 'Reference Centroids',
                  'Transformed Centroids']
        if show_plots is True:
            plotting.do_centroid_plot(deepframe, [xdat_o1, xref_o1, xtrans_o1],
                                      [ydat_o1, yref_o1, ytrans_o1],
                                      [xdat_o2, xref_o2, xtrans_o2],
                                      [ydat_o2, yref_o2, ytrans_o2],
                                      [xdat_o3], [ydat_o3], labels=labels)
        else:
            pass

    return transform


def run_stage3(results, deepframe, out_frames, save_results=True,
               show_plots=False, root_dir='./', force_redo=False,
               extract_method='box', specprofile=None, soss_estimate=None,
               soss_width=25, soss_transform=None):
    # ============== DMS Stage 3 ==============
    # 1D spectral extraction.
    print('\n\n**Starting supreme-SPOON Stage 3**')
    print('1D spectral extraction\n\n')

    utils.verify_path(root_dir + 'pipeline_outputs_directory')
    outdir = root_dir + 'pipeline_outputs_directory/Stage3/'
    utils.verify_path(outdir)

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
            if use_applesoss is True:
                specprofile = specprofilestep(deepframe,
                                              save_results=save_results,
                                              output_dir=outdir)[1]
                specprofile = outdir + specprofile
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
    for i, segment in enumerate(results):
        expected_file = outdir + fileroots[i] + step_tag
        if expected_file in all_files and force_redo is False:
            print('Output file {} already exists.'.format(expected_file))
            print('Skipping 1D Extraction Step.')
            res = expected_file
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
            except Exception as err:
                if str(err) == '(m>k) failed for hidden m: fpcurf0:m=0':
                    if soss_estimate is None:
                        if i != 0:
                            atoca_spectra = outdir + fileroots[i-1] + 'AtocaSpectra.fits'
                            soss_estimate = utils.get_soss_estimate(atoca_spectra,
                                                                    output_dir=outdir)
                        else:
                            print('No completed segments to create soss_estimate.')
                            raise err

                    print('\nInitial flux estimate failed, trying again with soss_estimate.\n')
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
        new_results.append(res)
    results = new_results

    # ===== Lightcurve Construction Step =====
    # Custom DMS step.
    stellar_spectra = construct_lightcurves(results, out_frames=out_frames,
                                            output_dir=outdir,
                                            save_results=save_results,
                                            show_plots=show_plots,
                                            extract_params=extract_params)

    return stellar_spectra


if __name__ == "__main__":
    # =============== User Input ===============
    root_dir = './'  # Root directory.
    indir = root_dir + 'pipeline_outputs_directory/Stage2/'  # Stage 2 results directory.
    input_filetag = 'badpixstep'  # Stage 2 results filetag.
    deepframe_file = None  # Median stack file.
    extract_method = 'box'  # Extraction method, box or atoca.
    soss_width = 25  # For extraction, box width.
    specprofile = None  # For ATOCA extraction, reference trace profiles.
    soss_estimate = None  # For ATOCA extraction, soss_estimate model.
    out_frames = [50, -50]  # For lightcurve normalization, transit ingress and egress integrations.
    force_redo = False  # Force redo of completed steps.
    # ==========================================

    # Set the CRDS cache variables.
    import os
    os.environ['CRDS_PATH'] = root_dir + 'crds_cache'
    os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

    # Unpack all files in the input directory.
    input_files = utils.unpack_input_directory(indir, filetag=input_filetag,
                                               exposure_type='CLEAR')
    print('\nIdentified {} CLEAR exposure segment(s):'.format(len(input_files)))
    for file in input_files:
        print(' ' + file)

    # Unpack the median stack.
    deepframe = fits.getdata(deepframe_file, 0)

    # Run segments through Stage 3.
    stellar_spectra = run_stage3(input_files, deepframe=deepframe,
                                 save_results=True, show_plots=False,
                                 root_dir=root_dir, force_redo=force_redo,
                                 extract_method=extract_method,
                                 specprofile=specprofile,
                                 out_frames=out_frames,
                                 soss_estimate=soss_estimate,
                                 soss_width=soss_width)
