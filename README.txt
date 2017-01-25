SNR Modelling Program - version 1.0, January 2017
===================================================

Description: A GUI program used to model supernova remnants.


Python requirements
-------------------

Note: Users are encouraged to use the provided executable file for Windows if
an appropriate version of Python is not already installed.

Python version >= 3.5

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

For executable version: run snr.exe (located within WindowsVersion folder)

For Python version: assuming Python has been added to the PATH, run from the
command line using "python snr.py"

Required files (must remain in the same folder):
 - snr.py
 - snr_calc.py
 - snr_gui.py
 - snr_plot.py
 - data directory (should contain 11 files)


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
 - ED: ejecta-dominated
 - ST: Sedov-Taylor
 - PDS: pressure-driven snowplow
 - MCS: momentum-conserving shell
 - WL: White & Long model (cloudy ISM)
 - LK: Liang & Keilty model (fractional energy loss)
 - TW: Tang & Wang model (hot low-density media)

See arXiv paper (astrop-ph.HE 1701.05942) for details of models used and associated
calculations.

This program is open source and is licensed under The 3-Clause BSD License.