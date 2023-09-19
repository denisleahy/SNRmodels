SNR Modelling Program - version 2.0, Sept, 2022
===================================================
Developed by Dr. Denis Leahy, Bryson Lawton and Jacqueline Williams

Description: A GUI program used to model supernova remnants.

See published papers (A Python Calculator for Supernova Remnant Evolution, D. A. Leahy and J. E. Williams, ApJ 153, 239, 2017; Emission Measures and Emission-measure-weighted Temperatures of Shocked Interstellar Medium and Ejecta in Supernova Remnants, Denis Leahy, Yuyang Wang, Bryson Lawton, Sujith Ranasinghe, and Miroslav Filipović, ApJ 158, 149, 2019; Evolutionary Models for 43 Galactic Supernova Remnants with Distances and X-Ray Spectra, D. A. Leahy , S. Ranasinghe, and M. Gelowitz, ApJSupp. 248, 16) for details of models used and associated
calculations. 

Python requirements
-------------------

Note: Users are encouraged to use the provided executable file for Windows if
an appropriate version of Python is not already installed.

Python version >= 3.6

Required libraries:
 - NumPy
 - SciPy
 - Matplotlib

The Anaconda distribution of Python (https://www.continuum.io/downloads) comes
with these libraries; however some distributions have a broken version of
Matplotlib. If the program does not run, try uninstalling Matplotlib and 
replacing it using pip.


How to run
----------

For Python version: assuming Python has been added to the PATH, run from the
command line using "python snr.py"

Required files (must remain in the same folder):
 - snr.py
 - snr_calc.py
 - snr_fileCalc.py
 - snr_gui.py
 - snr_plot.py
 - data directory (should contain 30 files)


Program usage details
---------------------

Program takes supernova remnant (SNR) properties as user inputs and produces
the following outputs:
 - Values at user-specified time: SNR radius, velocity, electron temperature
 - SNR phase transition times
 - Plots of SNR radius and velocity as functions of time

For some models, an additional set of outputs are produced by clicking the
"Emissivity" button. These outputs include:
 - Emission measure
 - Emission weighted temperature
 - Total luminosity over a user-specified energy range
 - Plots: 
    - Temperature as a function of radius
    - Density as a function of radius
    - Specific intensity as a function of impact parameter
    - Luminosity as a function of energy
    
Abbreviations used:
 - RS: reverse shock
 - ED: ejecta-dominated
 - ST: Sedov-Taylor
 - PDS: pressure-driven snowplow
 - MCS: momentum-conserving shell
 - CISM: Cloudy ISM phase, starts at t_ST (defined in TM99)

-----------------------------------
Version 1.1 Changes
-----------------------------------
- Time when the reverse shock reaches the core's edge now displayed under transition times
- Time when the reverse shock reaches the center now displayed under transition times
- s=0 case now expanded to address the n=1,9,11,13 cases
- s=0 case constants replaced with more accurate ones
- s=2 case now expanded to address the n= 6,8,9,10,11,12,13,14 cases
- s=2 case had n=0,1,2,4 cases removed while solutions are improved
- Various fixes with the emissivity button for various models
- Fractional Energy Loss model input fixed
- Standard model data files updated with more accurate structure models from the Chevalier & Parker solutions
- Cloudy ISM model data files updated with more accurate structure models from White & Long solutions
- 5th model type "Sedov-Taylor" with zero ejecta mass added
- Various smaller bug fixes

-----------------------------------
Version 1.1.1 Changes
-----------------------------------
- Emission Measure and Temperature plots added to main window for s = 0, n > 5 cases

-----------------------------------
Version 2.0 Changes
----------------------------------

See "Evolutionary Models for 43 Galactic Supernova Remnants with Distances and X-Ray Spectra, D. A. Leahy , S. Ranasinghe, and M. Gelowitz, ApJSupp. 248, 16" for the inverse mode calculation, added to SNRpy Version 2.0. 

This program is open source and is licensed under The 3-Clause BSD License.