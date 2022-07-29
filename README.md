# supreme-SPOON
supreme-**S**teps to **P**rocess S**O**SS **O**bservatio**N**s

JWST NIRISS SOSS pipeline.

### Instructons
The supreme-SPOON pipeline can be run from Stages 1 (detector level processing) to 3 (1D extraction) 
via the ```run_DMS.py``` script with the following steps:

1. Copy the ```run_DMS.py``` script into your working directory.
2. Fill in the "User Input" section, the critical inputs are summarized below:

    Stage 1 Inputs
   - root_dir : root directory from which all relative paths will be referenced.
   - uncal_indir : directory with the uncalibrated SOSS data.
   - input_filetag : file name tag given to uncalibrated data. Data from MAST should have the "uncal" flag.

    Stage 2 Inputs
   - background_file : path to a SOSS background model. It is recommended to use the ones provided by STScI, found here: https://jwst-docs.stsci.edu/jwst-calibration-pipeline-caveats/jwst-time-series-observations-pipeline-caveats/niriss-time-series-observation-pipeline-caveats#NIRISSTimeSeriesObservationPipelineCaveats-SOSSskybackground
   
    Other Parameters
   - save_results : if True, save intermediate results of each step.
   - process_f277w : if True, also process any F277W exposures present in the dataset.
   - extract_method : "box" or "atoca" - runs the applicable 1D extraction method. 
   - out_frames : Indices of intigrations corresponding to the beginning and end of the transit.
   
3. Once happy with the input parameters, enter ```python run_DMS.py``` in the terminal.



