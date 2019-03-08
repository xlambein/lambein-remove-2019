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

from joblib import Parallel, delayed
import multiprocessing

from datasets import loader
# from tqdm import tqdm

from grmf import GRMF
from pca import PCA

from util import cube_collapse, cube_expand, annulus_mapping, flux_cube, stim_cube, snr_cube, flux_loglr_path
from perf_assess import _fit_score, pixels_in_annulus, _fit_complete_score_parallel, frame_crop_annulus, noise_per_sep


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
    

def run(
    dataset='bpic_naco',
    cube_size=50, tscale=1., xscale=1.,
    sep_from=1., sep_to=4.,
    n_jobs=None, verbose=10,
    prad=1., model=None, opts=dict(),
    planets=[], planets_flevel_noise=False,
):
    dataset_name = dataset
    model_name = model
    
    if n_jobs is None:
        n_jobs = max(1, (multiprocessing.cpu_count() - 4)/THREADS_LIMIT)
    
    if model is None:
        print("Please provide a model")
        exit(1)
    
    
    ### MODEL LOADING ###

    model = globals()[model_name](**opts)
    
    matrix_completion = ('complete' in model.__class__.__dict__)
    
    
    ### DATASET LOADING ###
    
    # Load data
    dataset = loader(dataset_name)
    print("Loaded dataset {}".format(dataset_name))
    
    # Rescale, possibly
    if xscale != 1. or tscale != 1.:
        dataset = dataset.resample(spatial=xscale, temporal=tscale)
        print("Cube resampled by factors {} (spatially) and {} (temporally)".format(xscale, tscale))
    
    # Resize
    dataset.crop_frames(cube_size)
    print("Cube resized to ({}, {})".format(*dataset.cube.shape[1:]))
    
    
    ### INJECT PLANETS, MAYBE ###
    
    if len(planets) > 0:
        noise = noise_per_sep(dataset, [sep_from, sep_to])
    
    for planet in planets:
        sep, angle, flevel = planet
        if planets_flevel_noise:
            flevel = flevel * noise(sep)
        print("Injecting companion at distance {:.2f} FWHM, angle {:.1f} deg, flevel {:.2f}".format(
            sep, angle, flevel))
        dataset = dataset.inject_companion(sep*dataset.fwhm, angle, flevel)
    
    
    ### WRITING HEADER ###

    params = dict(
        dataset=dataset_name,
        cube_size=cube_size,
        tscale=tscale,
        xscale=xscale,
        planets=planets,
        planets_flevel_noise=planets_flevel_noise,
        
        sep_from=sep_from,
        sep_to=sep_to,
        
        prad=prad,
        model=model_name,
        opts=opts,
        matrix_completion=matrix_completion,
    )
    
    output_filename = 'perf/maps_{}_{}.pkl'.format(thetime, params['model'])
    print("Writing results to \"{}\"".format(output_filename))
    
    with open(output_filename, "a+") as f:
        pickle.dump(params, f)
    
    
    ### DATA COLLECTION ###
    
    start_time = time.time()

    print()
    print("Starting data collection on loaded dataset, with parameters:")
    print(" - Annulus {}--{} (FWHM)".format(sep_from, sep_to))
    print(" - Model \"{}\"{}".format(
        type(model).__name__,
        " (matrix completion with radius {} FWHM)".format(prad)
        if matrix_completion else ""
    ))
    print("   with parameters: {}".format(
        ', '.join('{}={}'.format(k, repr(v)) for (k, v) in opts.iteritems())
    ))
    print(" - Nbr of parallel processes {}".format(n_jobs))
    print()
    
    cy, cx = vip.var.frame_center(dataset.cube[0])
    fwhm = dataset.fwhm
    
    if not matrix_completion:
        
        scoring_functions = {
            'flux': flux_cube,
            'stim': stim_cube,
            'snr': snr_cube,
        }
        
        score_maps = _fit_score(dataset, model, scoring_functions=scoring_functions)
    
    else:
        
        scoring_functions = {
            ('flux', 'loglr'): flux_loglr_path,
        }
        
        score_maps = _fit_complete_score_parallel(
            dataset, model, prad,
            pixels_in_annulus(dataset.cube.shape[1:], (cy, cx), sep_from*fwhm, sep_to*fwhm),
            parallel=Parallel(n_jobs=n_jobs, verbose=verbose),
            scoring_functions=scoring_functions
        )

    # Flatten the score maps
    for key, val in score_maps.items():
        if type(key) == tuple:
            for name, scores in zip(key, val):
                score_maps[name] = scores
    for key in score_maps.keys():
        if type(key) == tuple:
            del score_maps[key]

    for name in score_maps.keys():
        score_maps[name] = frame_crop_annulus(score_maps[name], (cy, cx), sep_from*fwhm, sep_to*fwhm)
    
    with open(output_filename, "a+") as f:
        pickle.dump(score_maps, f)
    
    stop_time = time.time()
    
    print()
    print("Finished")
    print("Elapsed time: {}".format(hms_string(stop_time-start_time)))


if __name__ == '__main__':
    
    thetime = time.strftime("%Y%m%d-%H%M%S")
    
    ### CONFIGURATION ###
    
    # Default params
    params = dict(
        dataset='bpic_naco',
        cube_size=50,
        tscale=1.,
        xscale=1.,

        sep_from=1.,
        sep_to=4.,
        prad=1.,
    )
    
    # Read from STDIN if available
    if len(sys.argv) > 1:
        if sys.argv[1] == 'stdin':
            p = pickle.load(sys.stdin)
            params.update(p)
    
    # Otherwise, use local config
    else:
    
        ### CHANGE STUFF HERE ###
        params['dataset'] = 'sphere_k1_51eri_2016'
        # params['dataset'] = 'bpic_naco'
        
        params['cube_size'] = 100
        params['sep_from'] = 1.
        params['sep_to'] = 10.
        # params['planets_flevel_noise'] = True
        # params['planets'] = [(1.5, 120, 1.), (2., 240, 1.), (2.5, 0, 1.)]

        params['n_jobs'] = 10

        # params['model'] = 'GRMF'
        params['model'] = 'PCA'
        params['opts'] = dict(
            # rank=17,
            rank=15,
            # spatial_graph='corr', xlen=2., lx=160.,
            # temporal_graph='corr', tlen=5., lt=20.,
            # spatial_graph='nonlocal', xthresh=.4, xsig=.4, lx=100.,
            threads=2, maxit=20,
            # threads=2, maxit=10,
        )

        #########################
    
    run(**params)

