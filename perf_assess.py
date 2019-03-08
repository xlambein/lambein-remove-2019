import numpy as np
import scipy
import pandas as pd
# import sklearn
from sklearn import metrics
import vip_hci as vip
from joblib import Parallel, delayed
import multiprocessing
import pickle
# import threading
from tqdm import tqdm

from util import flux_cube, stim_cube, snr_cube, loglr_path_cube, flux_loglr_path, cart2rad, trajectory_mask
from copy import deepcopy


### UTILITY FUNCTIONS ###

def disk_bounding_box(center, radius):
    y, x = center
    
    ry = np.sqrt(radius*radius - (x - int(x))**2)
    rx = np.sqrt(radius*radius - (y - int(y))**2)
    
    t = int(np.ceil(y - ry))
    b = int(np.floor(y + ry))+1
    l = int(np.ceil(x - rx))
    r = int(np.floor(x + rx))+1
    
    return (t, l, b, r)


def mask_annulus(shape, center, inner_radius, outer_radius):
    cy, cx = center
    
    ys, xs = np.indices(shape)
    return ((ys - cy)**2 + (xs - cx)**2 <= outer_radius**2) &\
           ((ys - cy)**2 + (xs - cx)**2 >= inner_radius**2)


def pixels_in_annulus(shape, center, inner_radius, outer_radius):
    ys, xs = np.indices(shape)
    mask = mask_annulus(shape, center, inner_radius, outer_radius)
    
    return ys[mask], xs[mask]


def pixels_in_disk(shape, center, radius):
    cy, cx = center
    
    ys, xs = np.indices(shape)
    mask = (ys - cy)**2 + (xs - cx)**2 <= radius**2
    
    return ys[mask], xs[mask]


def frame_crop_disk(img, center, radius):
    cy, cx = center
    t, l, b, r = disk_bounding_box(center, radius)
    t, l = max(t, 0), max(l, 0)
    b, r = min(b, img.shape[0]), min(r, img.shape[1])
    
    img = img[t:b, l:r].copy()
    
    ys, xs = np.indices(img.shape)
    mask = (ys - cy + t)**2 + (xs - cx + l)**2 > radius**2
    img[mask] = np.nan
    
    return img


def frame_crop_annulus(img, center, inner_radius, outer_radius):
    cy, cx = center
    t, l, b, r = disk_bounding_box(center, outer_radius)
    t, l = max(t, 0), max(l, 0)
    b, r = min(b, img.shape[0]), min(r, img.shape[1])
    
    img = img[t:b, l:r].copy()
    
    ys, xs = np.indices(img.shape)
    mask = ((ys - cy + t)**2 + (xs - cx + l)**2 > outer_radius**2) |\
           ((ys - cy + t)**2 + (xs - cx + l)**2 < inner_radius**2)
    img[mask] = np.nan
    
    return img


def mean_on_mask(res, mask, **kwargs):
    return res[mask].mean()


def mean_l1_error_on_mask(res, mask, **kwargs):
    return np.abs(res[mask]).mean()


def mean_l2_error_on_mask(res, mask, **kwargs):
    return np.sqrt((res[mask]**2).mean())

### DATA COLLECTION ###


def _fit_score(
    dataset, model,
    scoring_functions
):
    res = dataset.cube - model.fit(dataset)
    res_d = vip.preproc.cube_derotate(res, dataset.angles)
    
    return {
        name: scoring_function(res=res, res_d=res_d, fwhm=dataset.fwhm)
        for name, scoring_function in scoring_functions.items()
    }


def _complete_score_at(
    pos, dataset, model, prad,
    scoring_functions
):
    cy, cx = vip.var.frame_center(dataset.cube)
    y, x = pos
    y, x = y - cy, x - cx
    rad, theta = cart2rad(y, x)
    
    mask = trajectory_mask(dataset, rad, theta, prad)

    res = dataset.cube - model.complete(dataset, mask)
    res_d = vip.preproc.cube_derotate(res, dataset.angles)
    
    return {
        name: scoring_function(res=res, res_d=res_d, psfn=dataset.psfn, angles=dataset.angles, fwhm=dataset.fwhm, rad=rad, theta=theta, mask=mask, pos=pos, prad=prad)
        for name, scoring_function in scoring_functions.items()
    }


def _fit_complete_score(
    dataset, model, prad, pixels,
    scoring_functions
):
    model.fit(dataset)
    
    score_maps = {
        name: tuple(np.zeros(dataset.cube.shape[1:]) for _ in name) if type(name) == tuple else np.zeros(dataset.cube.shape[1:])
        for name in scoring_functions.keys()
    }
    
    for (y, x) in zip(*pixels):
        scores = _complete_score_at((y, x), dataset, model, prad, scoring_functions=scoring_functions)
        for name, score in scores.items():
            if type(name) == tuple:
                for score_map, s in zip(score_maps[name], score):
                    score_map[y, x] = s
            else:
                score_maps[name][y, x] = score
    
    return score_maps


def _fit_complete_score_parallel(
    dataset, model, prad, pixels, parallel,
    scoring_functions
):
    model.fit(dataset)
    
    results = parallel(
        delayed(_complete_score_at)(pixel, dataset, model, prad, scoring_functions=scoring_functions)
        for pixel in zip(*pixels)
    )

    score_maps = {
        name: tuple(np.zeros(dataset.cube.shape[1:]) for _ in name) if type(name) == tuple else np.zeros(dataset.cube.shape[1:])
        for name in scoring_functions.keys()
    }
    for (scores, y, x) in zip(results, *pixels):
        for name, score in scores.items():
            if type(name) == tuple:
                for score_map, s in zip(score_maps[name], score):
                    score_map[y, x] = s
            else:
                score_maps[name][y, x] = score
    
    return score_maps


def _inject_fit_score(dataset, model, prad, pos, flevel, matrix_completion,
                      scoring_functions,
                      output_queue=None):
    cy, cx = vip.var.frame_center(dataset.cube[0])
    fwhm = dataset.fwhm
    
    y, x = pos
    dist, angle = cart2rad(y, x)
    sep = dist / fwhm

    fake_dataset = deepcopy(dataset)
    fake_dataset.inject_companions(flevel, rad_dists=dist, theta=angle, verbose=False)
    
    if not matrix_completion:

        score_maps = _fit_score(fake_dataset, model, scoring_functions=scoring_functions)

    else:
        
        score_maps = _fit_complete_score(
            fake_dataset, model, prad,
            pixels_in_disk(dataset.cube.shape[1:], (cy + y, cx + x), fwhm/2),
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
        score_maps[name] = frame_crop_disk(score_maps[name], (cy + y, cx + x), fwhm/2)

    positive = dict(
        x=x,
        y=y,
        sep=sep,
        dist=dist,
        angle=angle,
        flevel=flevel,
    )
    positive.update(score_maps)
    
    # Write output to file, if `output` is set
    if output_queue is not None:
        output_queue.put(positive)
    
    return positive


def noise_per_sep(dataset, seps):
    cube = dataset.cube
    fwhm = dataset.fwhm
    center = vip.var.frame_center(cube[0])
    
    seps[1] - seps[0]
    seps = np.linspace(seps[0], seps[1], int(np.ceil((seps[1] - seps[0]) * fwhm)))
    noise_levels = np.zeros(seps.shape)
    for i, sep in enumerate(seps):
        mask = mask_annulus(cube.shape[1:], center, sep*fwhm-fwhm/2, sep*fwhm+fwhm/2)
        noise_levels[i] = cube[:, mask].std()
    
    return scipy.interpolate.interp1d(seps, noise_levels, kind='linear', bounds_error=True)


def perf_assess(dataset, n_samples, seps, flevels, model, prad=1., flevel_noise_coef=1.,
                n_jobs=1, verbose=0, output_queue=None,
                random_state=None):
    """
    Performance assessment of a direct detection method.
    """
    if type(random_state) != np.random.RandomState:
        random_state = np.random.RandomState(random_state)
    
    if flevels == 'noise':
        noise = noise_per_sep(dataset, seps)
    elif type(flevels) == str:
        with open(flevels) as f:
            df = pd.read_csv(f)
        flevels = scipy.interpolate.interp1d(df.sep, df[['flux_min', 'flux_max']].as_matrix(), axis=0,
                                             kind='linear', fill_value='extrapolate')
    
    cy, cx = vip.var.frame_center(dataset.cube[0])
    fwhm = dataset.fwhm
    
    # A MC model is considered to own a `complete` method
    matrix_completion = ('complete' in model.__class__.__dict__)
    
    # Step 1: Compute the detection map of the empty cube
    if verbose > 0:
        print("Starting data collection for negatives")
    
    if not matrix_completion:
        
        scoring_functions = {
            'flux': flux_cube,
            'stim': stim_cube,
            'snr': snr_cube,
            'loglr_path': loglr_path_cube,
        }
        
        score_maps = _fit_score(dataset, model, scoring_functions=scoring_functions)
    
    else:
        
        scoring_functions = {
            ('flux', 'loglr'): flux_loglr_path,
            'err1': mean_l1_error_on_mask,
            'err2': mean_l2_error_on_mask,
        }
        
        score_maps = _fit_complete_score_parallel(
            dataset, model, prad,
            pixels_in_annulus(dataset.cube.shape[1:], (cy, cx), seps[0]*fwhm, seps[1]*fwhm),
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
        score_maps[name] = frame_crop_annulus(score_maps[name], (cy, cx), seps[0]*fwhm, seps[1]*fwhm)
    
    negatives = score_maps
    
    # Write negatives to file
    if output_queue is not None:
        output_queue.put(negatives)
    
    # Step 2: Inject planets in the cube and compute their local detection map
    if verbose > 0:
        print("Starting data collection for positives")
    
    pixels = np.array(zip(*pixels_in_annulus(dataset.cube.shape[1:], (cy, cx), seps[0]*fwhm, seps[1]*fwhm)))
    positions = pixels - np.array([cy, cx])
    positions = positions[random_state.choice(len(positions), size=n_samples)]
    if flevels == 'noise':
        fluxes = noise([cart2rad(*pos)[0]/fwhm for pos in positions]) * flevel_noise_coef
    elif callable(flevels):
        fluxes = [random_state.uniform(*flevels(cart2rad(*pos)[0]/fwhm)) for pos in positions]
    else:
        fluxes = random_state.uniform(*flevels, size=n_samples)

    positives = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_inject_fit_score)(dataset, model, prad, pos, flevel, matrix_completion,
                                   scoring_functions,
                                   output_queue=output_queue)
        for (pos, flevel) in zip(positions, fluxes)
    )
    
    return negatives, positives


def _frame_negatives_positives(img, positions, fwhm):
    """
    
    Notes
    -----
    Positions are wrt the center.
    """
    cy, cx = vip.var.frame_center(img)
    ys, xs = np.indices(img.shape)
    
    img = img.copy()
    
    pos_scores = []
    for pos in positions:
        y, x = pos
        mask = (ys - (y+cy))**2 + (xs - (x+cx))**2 < (fwhm/2)**2
        pos_scores.append(np.nanmax(img[mask]))
        erase_disk(img, (y+cy, x+cx), fwhm/2)
    
    negatives = frame_detections(img, fwhm)
    
    return negatives, pos_scores


def _inject_fit_detect_gomez2017(
    dataset, model, seps, pos, flevel
):
    cy, cx = vip.var.frame_center(dataset.cube[0])
    fwhm = dataset.fwhm
    
    y, x = pos
    dist, angle = cart2rad(y, x)
    sep = dist / fwhm

    fake_dataset = deepcopy(dataset)
    fake_dataset.inject_companions(flevel, rad_dists=dist, theta=angle, verbose=False)

    snr = _fit_score(fake_dataset, model, scoring_functions={'snr': snr_cube})
    snr = frame_crop_annulus(snr['snr'], (cy, cx), seps[0]*fwhm, seps[1]*fwhm)
    
    negatives, pos_scores = _frame_negatives_positives(snr, [(y, x)], fwhm)
    
    positive = dict(
        x=x,
        y=y,
        sep=sep,
        dist=dist,
        angle=angle,
        flevel=flevel,
        score=pos_scores[0]
    )
    
    return negatives, pd.Series(positive)


def perf_assess_gomez2017(
    dataset, n_samples, seps, flevels, model,
    n_jobs=1, verbose=0,
    random_state=None
):
    """
    Performance assessment of a direct detection method with the pipeline
    described in Gomez Gonzalez et al. 2017.
    """
    if type(random_state) != np.random.RandomState:
        random_state = np.random.RandomState(random_state)
    
    cy, cx = vip.var.frame_center(dataset.cube[0])
    fwhm = dataset.fwhm
    
    pixels = np.array(zip(*pixels_in_annulus(dataset.cube.shape[1:], (cy, cx), seps[0]*fwhm, seps[1]*fwhm)))
    positions = pixels - np.array([cy, cx])

    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_inject_fit_detect_gomez2017)(dataset, model, seps, pos, flevel)
        for (pos, flevel) in zip(
            positions[random_state.choice(len(positions), size=n_samples)],
            random_state.uniform(*flevels, size=n_samples)
        )
    )
    
    negatives = pd.concat([negative for (negative, _) in results], ignore_index=True)
    positives = pd.concat([positive for (_, positive) in results], axis=1).transpose()
    
    return negatives, positives


### DATA ANALYSIS ###


def load_pickle_results_old(filename):
    with open("../perf/" + filename) as f:
        return pickle.load(f)

    
def load_pickle_results(filename):
    with open("../perf/" + filename) as f:
        params = pickle.load(f)
        
        try:
            negatives = pickle.load(f)
        except EOFError:
            params, negatives, positives = load_pickle_results_old(filename)
        else:
            positives = []
            while True:
                try:
                    positives.append(pickle.load(f))
                except EOFError:
                    break
    
    if 'std' in negatives:
        negatives['flux_std'] = negatives['flux'] / negatives['std']
        for pos in positives:
            pos['flux_std'] = pos['flux'] / pos['std']
        
    return params, negatives, positives

    
def load_pickle_map(filename):
    with open("../perf/" + filename) as f:
        params = pickle.load(f)
        frames = pickle.load(f)
    
    return params, frames


def restrict_results_to_sep(results, seps, fwhm, negatives=False):
    params, negatives, positives = results
    
    par = params.copy()
    par['sep_from'], par['sep_to'] = seps

    if negatives:
        neg = {
            k: v.copy()
            for k, v in negatives.iteritems()
        }
    else:
        neg = {
            k: frame_crop_annulus(v, vip.var.frame_center(v), seps[0] * fwhm, seps[1] * fwhm)
            for k, v in negatives.iteritems()
        }

    pos = [
        p for p in positives
        if p['sep'] >= seps[0] and p['sep'] <= seps[1]
    ]
    
    return par, neg, pos


def stratify_results(results, sep_intervals, fwhm):
    return [restrict_results_to_sep(results, seps, fwhm) for seps in sep_intervals]


def erase_disk(img, pos, radius):
    y, x = pos
    ys, xs = np.indices(img.shape)
    
    smask = (ys - y)**2 + (xs - x)**2 <= radius**2
    img[smask] = np.nan
    
    return img


def soft_erase_disk(img, pos, psf):
    y, x = pos

    P = np.zeros_like(img)
    # P = vip.metrics.frame_inject_companion(P, psf, y, x, score / psf.max())
    stamp(P, psf * img[y, x] / psf.max(), (y, x))
    img -= P
    
    return img


def frame_detections(img, fwhm, min_radius=1.):
    cy, cx = vip.var.frame_center(img)
    ys, xs = np.indices(img.shape)
    
    img = img.copy()
    maxit = int(np.ceil(np.sum(~np.isnan(img)) / (np.pi * (fwhm/2)**2)))
    detections = []
    for _ in xrange(maxit):
        try:
            imax = np.nanargmax(img)
        except ValueError:
            break
        
        y, x = ys.flat[imax], xs.flat[imax]
        dist, angle = cart2rad(y-cy, x-cx)
        sep = dist / fwhm
        detections.append(dict(
            x=x-cx,
            y=y-cy,
            dist=dist,
            angle=angle,
            sep=sep,
            score=img[y, x],
        ))
        
        erase_disk(img, (y, x), min_radius * fwhm)
    
    return pd.DataFrame(detections)


def results_detections(results, fwhm, score_column=None, min_radius=1.):
    params, negatives, positives = results
    
    if score_column is None:
        if params['matrix_completion']:
            score_column = 'loglr'
        else:
            score_column = 'snr'
    
    positives = pd.DataFrame(positives)
    positives['score'] = positives[score_column].apply(np.nanmax)

    negatives = frame_detections(negatives[score_column], fwhm, min_radius)
#     negatives.rename(columns={'score': score_column}, inplace=True)
    
    return params, negatives, positives


def roc(positives, negatives, column=None):
    if type(positives) != pd.DataFrame:
        positives = pd.DataFrame(positives)
    if type(negatives) != pd.DataFrame:
        negatives = pd.DataFrame(negatives)
    if column is None:
        column = positives.columns[0]
    
    negatives = negatives[[column]].sort_values([column], ascending=[False])
    positives = positives[[column]].sort_values([column], ascending=[False])
    
    negatives['fp'] = 1.
    negatives['fp'] = negatives.fp.cumsum()
    negatives['fpr'] = negatives.fp / len(negatives)
    
    positives['tp'] = 1.
    positives['tp'] = positives.tp.cumsum()
    positives['tpr'] = positives.tp / len(positives)
    
    df = pd.concat([negatives[[column, 'fpr']], positives[[column, 'tpr']]])
    df = df.sort_values([column], ascending=[False], kind='mergesort')
    df = df.fillna(method='ffill').fillna(0.)
    
    return df.fpr.values, df.tpr.values


def roc_total_fp(positives, negatives, column=None):
    if type(positives) != pd.DataFrame:
        positives = pd.DataFrame(positives)
    if type(negatives) != pd.DataFrame:
        negatives = pd.DataFrame(negatives)
    if column is None:
        column = positives.columns[0]
    
    negatives = negatives[[column]].sort_values([column], ascending=[False])
    positives = positives[[column]].sort_values([column], ascending=[False])
    
    negatives['fp'] = 1.
    negatives['fp'] = negatives.fp.cumsum()
    
    positives['tp'] = 1.
    positives['tp'] = positives.tp.cumsum()
    positives['tpr'] = positives.tp / len(positives)
    
    df = pd.concat([negatives[[column, 'fp']], positives[[column, 'tpr']]])
    df = df.sort_values([column], ascending=[False], kind='mergesort')
    df = df.fillna(method='ffill').fillna(0.)
    
    # Stop when tpr reaches 1.
    df = df.iloc[:np.flatnonzero(df.tpr == 1.)[0]+1]
    
    return df.fp.values, df.tpr.values


def roc_mean_fp(positives, negatives, column=None):
    fp, tpr = roc_total_fp(positives, negatives, column)
    return fp / len(positives), tpr


def auc(x, y):
    return metrics.auc(x, y)


def aac(x, y):
    return metrics.auc(x, 1. - y)


def roc_auc(positives, negatives, column=None):
    return metrics.auc(*roc(positives, negatives, column))


def roc_aac(positives, negatives, column=None):
    fp, tpr = roc_total_fp(positives, negatives, column)
    return metrics.auc(fp, 1. - tpr)


def results_roc_aac(results, fwhm, score_column=None, min_radius=1.):
    det = results_detections(results, fwhm, score_column, min_radius=min_radius)
    res = pd.Series(det[0])

    fp, tpr = roc_total_fp(det[2], det[1], 'score')
    res['fp'] = fp
    res['tpr'] = tpr
    res['aac'] = metrics.auc(fp, 1. - tpr)
    
    return res


def auc_by(positives, negatives, by, column=None):
    return positives.groupby(by).apply(auc, negatives, column)


def auc_results(results, by=None, column=None):
    positives = results[results.companion == True]
    negatives = results[results.companion == False]
    if by is None:
        return auc(positives, negatives, column)
    else:
        return auc_by(positives, negatives, by, column)

    
def auc_results_old(results, column='stim'):
    results = results.sort_values([column], ascending=[False])

    injected = results[results.companion == True].copy()
    injected['tp'] = injected.groupby(['sep', 'flevel']).companion.cumsum()
    injected['tpr'] = injected.groupby(['sep', 'flevel']).tp.transform(lambda x: x / x.max())

    empty = results[results.companion == False].copy()
    empty.companion = ~empty.companion
    empty['fp'] = empty.groupby('sep').companion.cumsum()
    empty['fpr'] = empty.groupby('sep').fp.transform(lambda x: x / x.max())
    empty.companion = ~empty.companion
    
    def compute_auc(df, empty):
        empty = empty[empty.sep == df.iloc[0].sep]
        df = pd.concat([df, empty]).sort_values(column, kind='mergesort', ascending=False)
        df = df.fillna(method='ffill').fillna(0.)
        return metrics.auc(df.fpr.tolist(), df.tpr.tolist())

    aucs = injected.groupby(['sep', 'flevel']).apply(compute_auc, empty)
    return aucs.unstack(0)

