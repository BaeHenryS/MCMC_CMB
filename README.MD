# README

This repository contains scripts and data for cosmological parameter estimation using the Markov Chain Monte Carlo (MCMC) method. To replicate some of the works done in the presentation, please use MCMC_Tutorial.ipynb. 


## MCMC Folder Overview
The MCMC folder contains these files:

MCMC_performance.py: This implements MCMC with the Rosenbrock Likelihood to test the performance of just the MCMC algorithm.

MCMC_py_TT.py: This implements MCMC with the Planck TT likelihood.

MCMC_py.py: This implements MCMC with the Planck TT,TE,EE+lowE likelihood.

MCMC_Visualize.ipynb: This Jupyter notebook provides tools for visualizing the results of the MCMC method.

archive: This directory contains archived checkpoint files from previous MCMC runs.

checkpoints: This directory contains checkpoint files for the current MCMC run. The checkpoint files are used to save the state of the MCMC run and can be used to resume the run if it was interrupted.

data: This directory contains data files used in the MCMC analysis.

misc: This directory contains miscellaneous files used in the MCMC analysis.

