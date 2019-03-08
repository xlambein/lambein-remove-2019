from __future__ import print_function

import os, sys

THREADS_LIMIT = 1
os.environ['MKL_NUM_THREADS'] = str(THREADS_LIMIT)
os.environ['NUMEXPR_NUM_THREADS'] = str(THREADS_LIMIT)
os.environ['OMP_NUM_THREADS'] = str(THREADS_LIMIT)

import numpy as np
import pandas as pd
import sklearn.metrics
import time
import pickle

os.environ['MPLBACKEND'] = 'agg'
import vip_hci as vip

import multiprocessing

from datasets import loader
# from tqdm import tqdm

from grmf import GRMF
from pca import PCA

from util import cube_collapse, cube_expand, annulus_mapping
from perf_assess import perf_assess_gomez2017


def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    if h > 0:
        return "{}h{:>02}m{:>05.2f}s".format(h, m, s)
    elif m > 0:
        return "{:>02}m{:>05.2f}s".format(m, s)
    else:
        return "{:>05.2f}s".format(s)
    

def run_perf_assess(
    dataset='bpic_naco_empty',
    cube_size=50, tscale=1., xscale=1.,
    n_samples_per_res=5, sep_from=1., sep_to=4., flevel_from=50, flevel_to=200,
    seed=0, n_jobs=None,
    prad=1., model=None, opts=dict(),
):
    dataset_name = dataset
    model_name = model
    
    if n_jobs is None:
        n_jobs = max(1, (multiprocessing.cpu_count() - 4)/THREADS_LIMIT)
    
    if model is None:
        print("Please provide a model")
        exit(1)
    
    
    ### MODEL LOADING ###

    n_samples = int(n_samples_per_res * 4 * (sep_to**2 - sep_from**2))
    model = globals()[model_name](**opts)
    
    
    ### DATASET LOADING ###
    
    # Load data
    dataset = loader(dataset_name)
    print("Loaded dataset {}".format(dataset_name))
    
    # Rescale, possibly
    if xscale != 1. or tscale != 1.:
        dataset = dataset.resample(spatial=xscale, temporal=tscale)
        print("Cube resampled by factors {} (spatially) and {} (temporally)".format(xscale, tscale))
    
    # Resize
    dataset = dataset.resize(cube_size)
    print("Cube resized to ({}, {})".format(*dataset.cube.shape[1:]))
    
    
    ### WRITING HEADER ###

    params = dict(
        dataset=dataset_name,
        cube_size=cube_size,
        tscale=tscale,
        xscale=xscale,
        
        n_samples=n_samples,
        sep_from=sep_from,
        sep_to=sep_to,
        flevel_from=flevel_from,
        flevel_to=flevel_to,
        seed=seed,
        
        model=model_name,
        opts=opts,
    )
    
    output_filename = 'perf/{}_gomez2017_{}.pkl'.format(thetime, params['model'])
    print("Writing results to \"{}\"".format(output_filename))
    
    
    ### DATA COLLECTION ###
    
    start_time = time.time()

    print()
    print("Starting data collection on loaded dataset, with parameters:")
    print(" - Samples {}".format(n_samples))
    print(" - Annulus {}--{} (FWHM)".format(sep_from, sep_to))
    print(" - Injected flux level {}--{}".format(flevel_from, flevel_to))
    print(" - Model \"{}\"".format(
        type(model).__name__
    ))
    print("   with parameters: {}".format(
        ', '.join('{}={}'.format(k, repr(v)) for (k, v) in opts.iteritems())
    ))
    print(" - Random seed {}".format(seed))
    print(" - Nbr of parallel processes {}".format(n_jobs))
    print()
    
    negatives, positives = perf_assess_gomez2017(
        dataset, n_samples, [sep_from, sep_to], [flevel_from, flevel_to],
        model,
        random_state=seed,
        n_jobs=n_jobs, verbose=10
    )
    
    with open(output_filename, 'w') as f:
        pickle.dump(params, f)
        pickle.dump(negatives, f)
        pickle.dump(positives, f)
    
    stop_time = time.time()
    
    print()
    print("Finished")
    print("Elapsed time: {}".format(hms_string(stop_time-start_time)))


if __name__ == '__main__':
    
    thetime = time.strftime("%Y%m%d-%H%M%S")
    
    ### CONFIGURATION ###
    
    # Default params
    params = dict(
        dataset='bpic_naco_empty',
        cube_size=50,
        tscale=1.,
        xscale=1.,

        n_samples_per_res=5,
        sep_from=1.,
        sep_to=4.,
        flevel_from=50,
        flevel_to=200,
        seed=0,
    )
    
    # Read from STDIN if available
    if len(sys.argv) > 1:
        if sys.argv[1] == 'stdin':
            p = pickle.load(sys.stdin)
            params.update(p)
    
    # Otherwise, use local config
    else:
    
        ### CHANGE STUFF HERE ###
        params['opts'] = dict(
            rank=17
        )
        params['model'] = 'PCA'
        #########################
    
    run_perf_assess(**params)
    
