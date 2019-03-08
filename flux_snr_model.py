from __future__ import print_function

import os, sys

import numpy as np
import pandas as pd
import sklearn.metrics
import scipy
import time
import pickle

os.environ['MPLBACKEND'] = 'agg'
import vip_hci as vip

from datasets import loader
from tqdm import tqdm

from pca import PCA

from util import rad2cart
    

def inject_fit_subtract_snr(dataset, model, dist, angle, flevel):
    cy, cx = vip.var.frame_center(dataset.cube[0])
    y, x = rad2cart(dist, angle)
    y, x = cy + y, cx + x
    fwhm = dataset.fwhm

    fake_dataset = dataset.inject_companion(dist, angle, flevel)
    
    res = fake_dataset.cube - model.fit(fake_dataset)
    res_d = vip.preproc.cube_derotate(res, fake_dataset.angs)
    
    pf = np.median(res_d, axis=0)
    
    snr = vip.metrics.snr.snr_ss(pf, (x, y), fwhm)
    
    return snr


def flux_snr_regression(dataset, cube_model, sep, snr_range, n_samples=100, warmup=10, margin=15, random_state=None):
    if warmup >= n_samples:
        raise ValueError("`warmup` should be smaller than `n_samples`")
    if margin < 1:
        raise ValueError("`margin` should be greater than 1")
    
    if type(random_state) != np.random.RandomState:
        random_state = np.random.RandomState(random_state)
    snr_range = np.array(snr_range)
    
    # Initial max flux chosen as the greatest value in median-subtracted cube
    flux_range = np.array([
        0, 
        (dataset.cube - np.median(dataset.cube, axis=0)).max()
    ])

    fluxes = np.zeros(n_samples)
    snrs = np.zeros(n_samples)
    for i in tqdm(range(n_samples)):
        flux = random_state.uniform(*flux_range)
        angle = random_state.uniform(0, 360)
        dist = dataset.fwhm * (sep + random_state.uniform(-.5, .5))

        fluxes[i] = flux
        snrs[i] = inject_fit_subtract_snr(dataset, cube_model, dist, angle, flux)

        if i >= warmup:
            imin = max(0, min(i-margin, margin))
            model = scipy.stats.linregress(fluxes[imin:i+1], snrs[imin:i+1])
            
            if model.slope > 0:
                flux_range = (snr_range - model.intercept) / model.slope
                flux_range[0] = max(0, flux_range[0])
                # print(flux_range)
    
    return fluxes[margin:], snrs[margin:]


if __name__ == '__main__':
    dataset_name = sys.argv[1]
    dataset = loader(dataset_name).resize(50)
    cube_model = PCA(rank=17)
    
    random_state = np.random.RandomState(0)
    
    snr_range = (1, 2)
    seps = [1.5, 2, 2.5, 3, 3.5]
    
    fluxes_min, fluxes_max = np.zeros(len(seps)), np.zeros(len(seps))
    
    for i, sep in tqdm(enumerate(seps)):
        fluxes, snrs = flux_snr_regression(
            dataset, cube_model, sep, snr_range,
            n_samples=100, margin=15, random_state=random_state)
        
        model = scipy.stats.linregress(fluxes, snrs)
        flux_range = (snr_range - model.intercept) / model.slope
        flux_range[0] = max(0, flux_range[0])
        
        fluxes_min[i], fluxes_max[i] = flux_range
        print(flux_range)
    
    results = pd.DataFrame(dict(sep=seps, flux_min=fluxes_min, flux_max=fluxes_max))
    
    with open('flux_snr_model_{}.csv'.format(dataset_name), 'w') as f:
        results.to_csv(f, index=False)

        