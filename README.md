
# Remove, Complete and Detect: Matrix Completion for Direct Imaging of Exoplanets

Source code for the paper "Remove, Complete and Detect: Matrix Completion for
Direct Imaging of Exoplanets" by Lambein et al. 2019.


## Requirements

You'll need Python 2.7 (yes I'm sorry 😢) and Octave (again, sorry) to run the scripts in this repository.

To set up everything, first you'll want to install and build the git submodule
for GRMF:

```bash
git submodule init
git submodule update
cd lib/grmf/exp-grmf-nips15
octave-cli --eval "install"
cd ../../../
```

After that, in a fresh Python virtualenv, install the requirements with:

```bash
pip install -r requirements.txt
```

Then you should be good to go!


## Directory Structure

	.
	├── README.md
	│	This file
	│	
	├── __init__.py
	│	To make this a Python package
	│	
	├── compute_processed_frame.py
	│	Script to compute a processed frame in parallel
	│	
	├── datasets.py
	│	Data structures for the datasets used in this paper
	│	
	├── dummy_models.py
	│	Dummy non-completion models for test
	│	
	├── flux_snr_model.py
	│	Script to compute the flux range for fake companion injection from the
	│	target SNR range of PCA
	│	
	├── grmf.py
	│	Matrix completion-based PSF estimation with graph-regularized matrix
	│	factorization (GRMF)
	│	
	├── pca.py
	│	The PCA PSF estimation technique, wrapped in a class that provides an
	│	interface consistent with other techniques
	│	
	├── perf_assess.py
	│	Utility functions for performance assessment
	│	
	├── perf_assess_gomez2017.py
	│	Performance assessment of a detection technique, as described in
	│	Gomez et al. 2017
	│	
	├── perf_assess_parallel.py
	│	Performance assessment of a detection technique, as described in this
	│	paper
	│	
	├── util.py
	│	Utility functions for matrix completion for PSF estimation
	│	
	└── viz.py
		Utility functions for visualization


## Contact

If you have any trouble running the code in this repo, or if you have
questions, don't hesitate to get in touch:

- by email: xlambein AT gmail
- on Twitter: [@xlambein](http://twitter.com/xlambein)

