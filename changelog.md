# Changelog
All notable changes to this project will be documented in this file.

### [0.1.0] -- 2022-08-29
#### Added
- Base prerelease code for Stages 1 - 3.

### [0.2.0] -- 2022-10-28
#### Added
- Updated run_DMS script to accept yaml config file.
- TracingStep can now calculate changes in x and y position of the trace as well as the FWHM.
- BackgroundStep has been moved to Stage 2. 
- Background is re-added after 1/f noise correction during Stage 1 such that it is properly treated by the non-linearity correction.
- Speed improvements for 1/f and hot pixel corrections.
- stage4.py: code including routines for light curve binning and parallelized fitting using ray and juliet (transit only currently).
- fit_lightcurves.py: script to fit spectrophotometric light curves.
- fit_lightcurves.yaml: config file for fit_lightcurves.py.

#### Removed
- Redundant plotting routines.
- Multiple depreciated utility functions.

### [0.3.0] -- 2022-12-22
#### Added
- Add time-domain jump detection for ngroup=2 observations.
- Allow for integration-level correction of 1/f noise.
- Paralellization of stability parameter calculations.
- Allow for piecewise subtraction of background.