# supreme-SPOON
supreme-**S**teps to **P**rocess S**O**SS **O**bservatio**N**s

**supreme-SPOON** is an end-to-end pipeline for NIRISS/SOSS time series observations (TSOs).
The pipeline is divided into four stages, broadly mirroring the official JWST DMS:
 - Stage 1: Detector Level Processing 
 - Stage 2: Spectroscopic Processing
 - Stage 3: 1D Spectral Extraction
 - Stage 4: Lightcurve Fitting (in development)
 
A major advantage of **supreme-SPOON** over other available NIRISS/SOSS pipelines is the ability to carry out end-to-end reductions (uncalibrated to atmosphere spectra) without relying on intermediate outputs from the JWST DMS.
Furthermore, **supreme-SPOON** is able to run the ATOCA extraction algorithm to explicitly model the order contamination that is known to affect SOSS observations. 

### Installation Instructions
The latest release of **supreme-SPOON** can be downloaded from PyPI by running:

    pip install supreme_spoon

or the latest development version can be grabbed from GitHub:

    git clone https://github.com/radicamc/supreme-spoon
    cd supreme_spoon
    pip install .

If you make use of this code in your work, please cite [Feinstein & Radica et al. (2022)](). 
If you use the ATOCA extraction algorithm, please also cite [Radica et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022arXiv220705136R/abstract) 
and [Darveau-Bernier et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022arXiv220705199D/abstract).

### Usage Instructions
The **supreme-SPOON** pipeline can be run in a similar fashion to the JWST DMS, by individually calling each step.
Alternatively, Stages 1 to 3 can be run at once via the ```run_DMS.py``` script with the following steps:

1. Copy the ```run_DMS.py``` script into your working directory.
2. Fill in the "User Input" section.
3. Once happy with the input parameters, enter ```python run_DMS.py``` in the terminal.
