#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Jul 21 17:33 2022

@author: MCR

Custom JWST DMS pipeline steps for Stage 3 (1D spectral extraction).
"""

from astropy.io import fits
import numpy as np
import warnings

from jwst import datamodels
from jwst.extract_1d.soss_extract import soss_solver
from jwst.pipeline import calwebb_spec2

from supreme_spoon import utils
from supreme_spoon import plotting


def construct_lightcurves(datafiles, output_dir, save_results=True,
                          show_plots=False, planet_name=None):
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
    t = utils.make_time_axis(file)
    out_trans = np.concatenate([np.arange(90), np.arange(40) - 40])

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
        plotting.plot_2dlightcurves(nflux_o1, nflux_o2, wave1d_o1, wave1d_o2)
        plotting.plot_2dlightcurves(nflux_o1 - nflux_o1_clip,
                                    nflux_o2 - nflux_o2_clip,
                                    wave1d_o1, wave1d_o2, **kwargs)
        plotting.plot_2dlightcurves(nflux_o1_clip, nflux_o2_clip, wave1d_o1,
                                    wave1d_o2)

    flux_o1_clip = nflux_o1_clip * norm_factor_o1
    flux_o2_clip = nflux_o2_clip * norm_factor_o2

    if save_results is True:
        # Save full res stellar spectra
        filename = output_dir + planet_name[:-1] + '_spectra_fullres.fits'
        header_dict, header_comments = utils.get_default_header()
        header_dict['Target_Name'] = planet_name[:-1]
        utils.write_spectra_to_file(filename, wave2d_o1, flux_o1_clip, ferr_o1,
                                    wave2d_o2, flux_o2_clip, ferr_o2, t,
                                    header_dict, header_comments)

        # Save full res lightcurves
        filename = output_dir + planet_name + '_lightcurves_fullres.fits'
        header_dict, header_comments = utils.get_default_header()
        header_dict['Target_Name'] = planet_name
        utils.write_spectra_to_file(filename, wave2d_o1, nflux_o1_clip,
                                    nferr_o1, wave2d_o2, nflux_o2_clip,
                                    nferr_o2, t, header_dict,
                                    header_comments)

    stellar_spectra = {'Wave 2D Order 1': wave2d_o1,
                       'Flux Order 1': flux_o1_clip,
                       'Flux Error Order 1': ferr_o1,
                       'Wave 2D Order 2': wave2d_o2,
                       'Flux Order 2': flux_o2_clip,
                       'Flux Error Order 2': ferr_o2,
                       'Time': t}
    normalized_lightcurves = {'Wave 2D Order 1': wave2d_o1,
                              'Flux Order 1': nflux_o1_clip,
                              'Flux Error Order 1': nferr_o1,
                              'Wave 2D Order 2': wave2d_o2,
                              'Flux Order 2': nflux_o2_clip,
                              'Flux Error Order 2': nferr_o2,
                              'Time': t}

    return normalized_lightcurves, stellar_spectra


def get_soss_transform(deepframe, datafile, show_plots=False):

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

    if show_plots is True:
        cen_o1, cen_o2, cen_o3 = utils.get_trace_centroids(deepframe,
                                                           'SUBSTRIP256')
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


def run_stage3():
    return


# TODO: Add main to just run stage 3
if __name__ == "__main__":
    # ===== User Input =====
    show_plots = False
    save_results = True
    planet_name = 'WASP-96b'
    # ======================

    run_stage3()

    # stage2_indir = 'pipeline_outputs_directory/Stage2/SecondPass/'
    # utils.verify_path('pipeline_outputs_directory/Stage3')
    # outdir = 'pipeline_outputs_directory/Stage3/'
    # res = utils.unpack_input_directory(stage2_indir, filetag='badpixstep',
    #                                    process_f277w=False)
    # clear_segments = res[0]
    # all_exposures = {'CLEAR': clear_segments}
    # print('\nIdentified {} CLEAR exposure segment(s):'.format(len(clear_segments)))
    # for file in clear_segments:
    #     print(' ' + file)
    #
    # for i, file in enumerate(clear_segments):
    #     data = fits.getdata(file, 1)
    #     if i == 0:
    #         cube = data
    #     else:
    #         cube = np.concatenate([data, cube], axis=0)
    # deepframe = np.nanmedian(cube, axis=0)
    #
    # # ===== 1D Extraction Step =====
    # transform = get_soss_transform(deepframe, clear_segments[0],
    #                                show_plots=show_plots)
    # results = []
    # for segment in clear_segments:
    #     step = calwebb_spec2.extract_1d_step.Extract1dStep()
    #     res = step.call(segment, output_dir=outdir, save_results=save_results,
    #                     soss_transform=[transform[0], transform[1],
    #                                     transform[2]],
    #                     soss_atoca=False, subtract_background=False,
    #                     soss_bad_pix='masking', soss_width=25,
    #                     soss_modelname=None)
    #     results.append(res)
    # # Hack to fix file names
    # results = utils.fix_filenames(results, 'badpixstep_', outdir)
    #
    # # ===== Construct Lightcurves =====
    # # Custom DMS step.
    # res = construct_lightcurves(results, output_dir=outdir,
    #                             save_results=save_results,
    #                             show_plots=show_plots,
    #                             planet_name=planet_name)
    # normalized_lightcurves, stellar_spectra = res
