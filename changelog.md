# Changelog
All notable changes to this project will be documented in this file.

### [0.1.0] -- 2022-08-29
#### Added
- Base prerelease code for Stages 1 - 3.

### [0.2.0] -- 2022-10-27
#### Added
- Code for Stage 4.
- Updated run_DMS script to accept yaml config file.
- TracingStep can now calculate changes in x and y position of the trace as well as the FWHM.
- BackgroundStep has been moved to Stage 2. 
- Background is re-added after 1/f noise correction during Stage 1 such that it is properly treated by the non-linearity correction.
- Speed improvements for 1/f and hot pixel corrections.

#### Removed
- Redundant plotting routines.
- Multiple depreciated utility functions.