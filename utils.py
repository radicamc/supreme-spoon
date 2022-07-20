from astropy.io import fits
import bottleneck as bn
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import warnings

from jwst import datamodels
from jwst.pipeline import calwebb_detector1
from jwst.pipeline import calwebb_spec2
from jwst.extract_1d.soss_extract import soss_boxextract

from sys import path

soss_path = '/home/radica/GitHub/jwst-mtl/'
path.insert(1, soss_path)

from SOSS.dms import soss_oneoverf
from SOSS.dms import soss_outliers
from SOSS.dms.soss_centroids import get_soss_centroids


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

    deepstack = bn.nanmedian(cube, axis=0)
    if return_rms is True:
        rms = bn.nanstd(cube, axis=0)
    else:
        rms = None

    return deepstack, rms




