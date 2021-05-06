# fast-few-body-bayesing

The corresponding arXiv publication can be found [here](https://arxiv.org/abs/2104.04441).

This package is built to calculate posterior probability distributions 
for three-body low-energy constants at next-to-next-to-leading order (NNLO) in chiral nuclear forces. 
Eigenvector continuation (EC/ Ritz Method) 
is used as an efficient emulator for few-body bound-state observable calculations to make both Markov Chain Monte Carlo
 (MCMC) sampling possible.

## Getting started

- This project relies on `python=3.8`. It was not tested with different versions.
  To view the entire list of required packages, see `environment.yaml`.
- Clone the repository to your local machine.
- Once you have `cd` into this repo, create a virtual environment (assuming you have `conda` installed) via
  ```bash
  conda env create -f environment.yml
  ```
- Enter the virtual environment with `conda activate fit-3bf`
- Install the `fit3bf` package in the repo root directory using `pip install -e .`
  (you only need the `-e` option if you intend to edit the source code in `/fit3bf`.

## Using your own chiral interaction

The inputs needed to use this package are Hamiltonians and operators for few-body systems. The workflow is the following

1. Package your Hamiltonian and operators for observables in the appropriate format. 
2. Train EC emulator for observables and output an emulator file for each observable.
3. Use these emulator files to efficiently sample posterior pdfs for $c_D$ and $c_E$. 

Following are details of how to perform these steps for your own chiral interaction to get posterior pdfs for $c_D$ and
$c_E$.

## Step 1: package operators in appropriate format

In order to train the EC emulators for the observables, you must construct files that the training script can use. This
will involve separating your operators into a constant part and any parts proportional to the LECs being sampled. Please
see reference for mathematical details and why this is necessary.

### Fitting cD and cE only (no pi-N or NN LEC uncertainty included)

The simplest case is the one in which you have optimal values for the pi-N and NN LECs, and wish to keep these fixed in
the analysis. If you have covariance matrices or posterior pdfs available for these other sectors and want to also 
include this information, please move to the next section.

### Fitting cD and cE (pi-N or NN LEC uncertainty included)

If you have covariance matrices or posterior pdfs available for the pi-N and NN sectors and want to include this
information when fitting cD and cE, the approach will be similar to the one above,
except that you should place EC training points near the likely values of these LECs as well.

## What to do next

Assuming you have followed the directions in the Getting Started section, to create the emulators used in this work:
1. Ensure the matrix element files are available (# TODO). The parameters files (discussed below) will have to point to the locations of these files.
1. `cd` to `scripts/` and then run `python train_emulators.py --parameters=parameters/NNLO_450.yaml`.
   This will generate emulators in the full `NN+3N` space of 13 dimensions.
   The output directory name will be unique based on the contents of the parameter file fed into `train_emulators.py`
1. Again while in `scripts/`, run `python train_emulators.py --parameters=parameters/NNLO_450_3bf_only.yaml`.
   This will generate emulators in the 2d `3N` space.

To sample the emulators (only done for the `NN+3N` emulators in this work) do the following:
again while in scripts run `python sample_emulators.py --parameters=parameters/sampling.yaml`,
ensuring the directory of the emulators matches what is in `sampling.yaml`.

To create the plots from these emulators and samples, check out the following in `notebooks/`:
- `plot_samples.ipynb`
- `latin_hypercube_samples.ipynb`
- `gridded_posteriors.ipynb`