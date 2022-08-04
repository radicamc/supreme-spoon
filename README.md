# supreme-SPOON
supreme-**S**teps to **P**rocess S**O**SS **O**bservatio**N**s

JWST NIRISS SOSS pipeline.

### Instructons
The supreme-SPOON pipeline can be run from Stages 1 (detector level processing) to 3 (1D extraction) 
via the ```run_DMS.py``` script with the following steps:

1. Copy the ```run_DMS.py``` script into your working directory.
2. Fill in the "User Input" section, the critical inputs are summarized below:

    Key Parameters
   - root_dir : root directory from which all relative paths will be referenced.
   - input_dir : directory with the input SOSS data.
   - input_filetag : file name tag given to input data. Uncalibarted data from MAST should have the "uncal" flag.
   
   Stage 1 Inputs
   - outlier_maps : list of paths to bad pixel maps to mask in 1/f noise correction.
   - trace_mask : path to trace mask for 1/f noise correction.
   - background_file : path to a SOSS background model. It is recommended to use the ones provided by STScI, found here: https://jwst-docs.stsci.edu/jwst-calibration-pipeline-caveats/jwst-time-series-observations-pipeline-caveats/niriss-time-series-observation-pipeline-caveats#NIRISSTimeSeriesObservationPipelineCaveats-SOSSskybackground
   
    Other Parameters
   - run_stages : list of pipeline stages to run.
   - save_results : if True, save intermediate results of each step.
   - exposure_type : "CLEAR" or "F277W". Processes only the corresponding exposure types.
   - extract_method : "box" or "atoca" - runs the applicable 1D extraction method. 
   - out_frames : Indices of integrations corresponding to the beginning and end of the transit.
   
3. Once happy with the input parameters, enter ```python run_DMS.py``` in the terminal.

**Note**: 1/f noise is an important consderation for SOSS observations, and incorrect treatment can led to large biases (diluton of transit depths, correlated noise, etc.) in the final extracted spectra.
To improve the estimation and correction of 1/f noise, it is adviseable to include a bad pixel, and a trace mask, particular to the dataset being analyzed, as well as an estimate of the transit curve as inputs to Stage 1. 
The bad pixel map is produced as part of Stage 1, and trace masks and light curve estimate are produced in Stage 2. They will have the file tags "dqpixelflags", "tracemask", and "lcscaling" respectively. It is therefore recommended to first compete a "trial" run of Stages 1 and 2 by specifying ```run_stages=[1, 2]```. 
The bad pixel maps, trace masks, and lightcurve estimate can then be included via the ```outlier_maps```, ```trace_mask```, and ```scaling_curve``` parameters respectively, and all three stages run.




