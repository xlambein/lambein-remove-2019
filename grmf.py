import numpy as np
import scipy
from oct2py import Oct2Py
import os
from util import interpolate_resample_cube, cube_collapse, cube_expand


def _get_octave():
    octave = Oct2Py()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    octave.addpath(os.path.join(current_dir, 'lib/grmf/exp-grmf-nips15/grmf-core/octave/'))
    
    return octave


_octave = None
_pid = None
def _get_process_octave():
    global _octave, _pid
    
    if _octave is None or _pid != os.getpid():
        _octave = _get_octave()
        _pid = os.getpid()
    
    return _octave


def grmf(mat, mask, rank=5, Lt=None, Lx=None, maxit=10, maxcgit=5, eps=.1, threads=1, output='normal', octave=None):
    """Matrix completion with graph-regularized matrix factorization (GRMF).
    
    This is an interface to run the GRMF solver in Octave through Python.
    
    For more information on the parameters of the model, see the documentation in:
    
        ./lib/grmf/exp-grmf-nips15/README
    
    as well as the GRMF paper:
    
        Rao, Nikhil et al. (2015). "Collaborative filtering with graph information:
        Consistency and scalable methods". In: Advances in neural information
        processing systems, pp. 2107-2115.
    
    Parameters
    ----------
    mat : np.array
        The data matrix, a 2D float array.
    
    mask : np.array
        The mask of the path in matrix form, a 2D boolean array.
    
    rank : int
        The rank of the model.
    
    Lt : sparse array or None
    Lx : sparse array or None
        The Laplacian matrix for the graph on the rows and columns, respectively.  If None, then an array
        of zeros is used (i.e. the graph is completely disconnected).
        
    maxit : int
        The maximum number of iterations.
    
    maxcgit : int
        The maximum number of iterations in a gradient descent step.
    
    eps : float
        The stopping criterion epsilon of tron.
    
    threads : int
        The number of threads used by the GRMF solver in Octave.
    
    output : str
        If 'debug', returns both the matrix completion output and additional debug information.
        If 'normal', only returns the matrix completion output.
    
    octave : Oct2Py
        The octave session in which to run GRMF.  It should be created with `_get_octave`,
        so as to have the necessary PATH.  If `None`, will create or re-use an existing session
        in a thread-safe manner.
    
    Returns
    -------
    np.array
        An approximation of `M` constructed with `mask` removed.
    
    rmse
        The root mean square error.
    walltime
        The wall clock time taken to run the algorithm.
    """
    
    # Get the process' Octave instance if one wasn't given
    if octave is None:
        octave = _get_process_octave()
    
    I, J = np.indices(mat.shape)[:, ~mask]
    Y = np.vstack([I+1, J+1, mat[I, J]]).T
    
    if output == 'debug':
        I, J = np.indices(mat.shape)[:, mask]
        Ytest = np.vstack([I+1, J+1, mat[I, J]]).T
    else:
        Ytest = []
    
    if Lt is None:
        Lt = scipy.sparse.csr_matrix((mat.shape[0], mat.shape[0]))
    
    if Lx is None:
        Lx = scipy.sparse.csr_matrix((mat.shape[1], mat.shape[1]))
    
    options = '-k {} -t {} -g {} -q {} -e {} -n {}'.format(rank, maxit, maxcgit, int(output == 'debug'), eps, threads)
    W, H, rmse, walltime = octave.glr_mf_train(Y, Ytest, Lt, Lx, options, nout=4, verbose=(output == 'debug'))
    Mh = np.dot(W.T, H)
    
    if output == 'normal':
        return Mh, W.T, H.T
    elif output == 'debug':
        return Mh, W.T, H.T, rmse, walltime
    else:
        raise ValueError("`output` should be 'normal' or 'debug'")


class GRMF(object):
    """
    An object, representing the GRMF matrix completion model.
    
    - `__init__()` creates a new model with specific parameters
    - `fit()` fits the model to a dataset
    - `complete()` creates a model PSF based on a dataset and a mask
    
    Note that `fit` must be called first before `complete`.
    """
    
    def __init__(
        self, rank=5, temporal_graph=None, spatial_graph=None,
        lt=1., lx=1., rt=1., rx=1.,
        tlen=None, tthresh=.5, tsig=1., tnorm=1,
        xlen=None, xthresh=.5, xsig=1., xnorm=1,
        maxit=10, maxcgit=5, eps=.1, threads=4
    ):
        """
        Notes
        -----
        Here, `xlen` is given in multiples of half-lambda/D.
        """
        _grmf_parameters_check(
            rank, temporal_graph, spatial_graph,
            lt, lx, rt, rx,
            tlen, tthresh, tsig, tnorm,
            xlen, xthresh, xsig, xnorm,
            maxit, maxcgit, eps, threads
        )
        
        self.rank = rank
        self.temporal_graph = temporal_graph
        self.spatial_graph = spatial_graph
        self.lt = lt
        self.lx = lx
        self.rt = rt
        self.rx = rx
        self.tlen = tlen
        self.tthresh = tthresh
        self.tsig = tsig
        self.tnorm = tnorm
        self.xlen = xlen
        self.xthresh = xthresh
        self.xsig = xsig
        self.xnorm = xnorm
        self.maxit = maxit
        self.maxcgit = maxcgit
        self.eps = eps
        self.threads = threads
        
    def fit(self, dataset, collapse=cube_collapse, expand=cube_expand):
        cube = dataset.cube
        angs = dataset.angles
        psf = dataset.psfn
        fwhm = dataset.fwhm
        mat = collapse(cube)
        
        if self.temporal_graph is None:
            Wt = scipy.sparse.csr_matrix((mat.shape[0], mat.shape[0]))

        elif self.temporal_graph == 'corr':
            # WARNING: Take note that the model is trained on the full cube, including the masked trajectory
            Wt = temporal_corr_weights(mat, angs, self.tlen)

        elif self.temporal_graph == 'nonlocal':
            # WARNING: Take note that the model is trained on the full cube, including the masked trajectory
            Wt = distance_weights(mat.T, thresh=self.tthresh, sig=self.tsig, norm=self.tnorm)


        if self.spatial_graph is None:
            Wx = scipy.sparse.csr_matrix((mat.shape[1], mat.shape[1]))

        elif self.spatial_graph == 'corr':
            # WARNING: Take note that the model is trained on the full cube, including the masked trajectory
            if self.xlen is not None:
                xlen = int(np.round(self.xlen * fwhm / 2))  # xlen in multiples of half-lambda/2
                psf = psf[psf.shape[0]/2-xlen:psf.shape[0]/2+xlen+1,
                          psf.shape[1]/2-xlen:psf.shape[1]/2+xlen+1]
            psf = np.maximum(0., psf) / psf.max()

            # TODO: Fix this when `collapse` is not just `cube_collapse`
            Wx = weights_from_corr_2d(psf, cube.shape[1:])

        elif self.spatial_graph == 'nonlocal':
            # Delayed to `complete`
            Wx = scipy.sparse.csr_matrix((mat.shape[1], mat.shape[1]))

        self.Lt = self.lt * weights_to_lap(Wt) + self.rt * scipy.sparse.eye(*Wt.shape)
        self.Lx = self.lx * weights_to_lap(Wx) + self.rx * scipy.sparse.eye(*Wx.shape)
    
    def complete(self, dataset, mask, collapse=cube_collapse, expand=cube_expand, octave=None, full_output=False):
        cube = dataset.cube
        mat = collapse(cube)
        mat_mask = collapse(mask)
        
        # Delayed computation of graph to use the mask
        if self.spatial_graph == 'nonlocal':
            Wx = distance_weights(
                mat, mat_mask, only_mask=True,
                thresh=self.xthresh, sig=self.xsig, norm=self.xnorm)
            self.Lx = self.lx * weights_to_lap(Wx) + self.rx * scipy.sparse.eye(*Wx.shape)
    
        mat_h, W, H = grmf(
            mat, mat_mask,
            rank=self.rank, Lt=self.Lt, Lx=self.Lx,
            maxit=self.maxit, maxcgit=self.maxcgit, eps=self.eps, threads=self.threads,
            octave=octave
        )
        if full_output:
            return expand(mat_h, cube), W, H
        else:
            return expand(mat_h, cube)


class GRMFNoCompletion(object):
    
    def __init__(
        self, rank=5, temporal_graph=None, spatial_graph=None,
        lt=1., lx=1., rt=1., rx=1.,
        tlen=None, tthresh=.5, tsig=1., tnorm=1,
        xlen=None, xthresh=.5, xsig=1., xnorm=1,
        maxit=10, maxcgit=5, eps=.1, threads=4
    ):
        """
        Notes
        -----
        Here, `xlen` is given in multiples of half-lambda/D.
        """
        _grmf_parameters_check(
            rank, temporal_graph, spatial_graph,
            lt, lx, rt, rx,
            tlen, tthresh, tsig, tnorm,
            xlen, xthresh, xsig, xnorm,
            maxit, maxcgit, eps, threads
        )
        
        self.rank = rank
        self.temporal_graph = temporal_graph
        self.spatial_graph = spatial_graph
        self.lt = lt
        self.lx = lx
        self.rt = rt
        self.rx = rx
        self.tlen = tlen
        self.tthresh = tthresh
        self.tsig = tsig
        self.tnorm = tnorm
        self.xlen = xlen
        self.xthresh = xthresh
        self.xsig = xsig
        self.xnorm = xnorm
        self.maxit = maxit
        self.maxcgit = maxcgit
        self.eps = eps
        self.threads = threads
        
    def fit(self, dataset, collapse=cube_collapse, expand=cube_expand, octave=None):
        cube = dataset.cube
        angs = dataset.angles
        psf = dataset.psfn
        fwhm = dataset.fwhm
        mat = collapse(cube)
        mat_mask = np.zeros(mat.shape, dtype=bool)
        
        if self.temporal_graph is None:
            Wt = scipy.sparse.csr_matrix((mat.shape[0], mat.shape[0]))

        elif self.temporal_graph == 'corr':
            # WARNING: Take note that the model is trained on the full cube, including the masked trajectory
            Wt = temporal_corr_weights(mat, angs, self.tlen)

        elif self.temporal_graph == 'nonlocal':
            # WARNING: Take note that the model is trained on the full cube, including the masked trajectory
            Wt = distance_weights(mat.T, thresh=self.tthresh, sig=self.tsig, norm=self.tnorm)


        if self.spatial_graph is None:
            Wx = scipy.sparse.csr_matrix((mat.shape[1], mat.shape[1]))

        elif self.spatial_graph == 'corr':
            # WARNING: Take note that the model is trained on the full cube, including the masked trajectory
            if self.xlen is not None:
                xlen = int(np.round(self.xlen * fwhm / 2))  # xlen in multiples of half-lambda/2
                psf = psf[psf.shape[0]/2-xlen:psf.shape[0]/2+xlen+1,
                          psf.shape[1]/2-xlen:psf.shape[1]/2+xlen+1]
            psf = np.maximum(0., psf) / psf.max()

            # TODO: Fix this when `collapse` is not just `cube_collapse`
            Wx = weights_from_corr_2d(psf, cube.shape[1:])

        elif self.spatial_graph == 'nonlocal':
            Wx = distance_weights_on_mask(
                mat, mat_mask, only_mask=True,
                thresh=self.xthresh, sig=self.xsig, norm=self.xnorm)

        self.Lt = self.lt * weights_to_lap(Wt) + self.rt * scipy.sparse.eye(*Wt.shape)
        self.Lx = self.lx * weights_to_lap(Wx) + self.rx * scipy.sparse.eye(*Wx.shape)
    
        mat_h, _, _ = grmf(
            mat, mat_mask,
            rank=self.rank, Lt=self.Lt, Lx=self.Lx,
            maxit=self.maxit, maxcgit=self.maxcgit, eps=self.eps, threads=self.threads,
            octave=octave
        )
        self.cube_h = expand(mat_h, cube)
        
        return self.model_psf()
    
#     def complete(self, dataset, mask):
#         return self.cube_h
    
    def model_psf(self):
        return self.cube_h


def _grmf_parameters_check(
    rank, temporal_graph, spatial_graph,
    lt, lx, rt, rx,
    tlen, tthresh, tsig, tnorm,
    xlen, xthresh, xsig, xnorm,
    maxit, maxcgit, eps, threads
):
    if temporal_graph is not None:
        if temporal_graph not in ('corr', 'nonlocal'):
            raise ValueError("`temporal_graph` should be `None`, 'corr' or 'nonlocal'")
    if spatial_graph is not None:
        if spatial_graph not in ('corr', 'nonlocal'):
            raise ValueError("`spatial_graph` should be `None`, 'corr' or 'nonlocal'")
    if temporal_graph == 'corr':
        if tlen is None:
            raise ValueError("'corr' temporal graph requires a temporal correlation length `tlen`")
    if type(maxit) != int:
        raise ValueError("`maxit` should be an integer number")
    if maxit <= 0:
        raise ValueError("`maxit` should be a positive integer")
    if type(maxcgit) != int:
        raise ValueError("`maxcgit` should be an integer number")
    if maxcgit <= 0:
        raise ValueError("`maxcgit` should be a positive integer")
    if eps <= 0:
        raise ValueError("`eps` should be a positive number")
    if type(threads) != int:
        raise ValueError("`threads` should be an integer number")
    if threads <= 0:
        raise ValueError("`threads` should be a positive integer")


def autocorrelation(mat):
    """Computes the average autocorrelation of several time series contained in a matrix.
    
    Considering `mat` to be an m-by-n matrix, we see it as a collection of n time series of length m.
    This function computes the autocorrelation of each of those time series, and then returns the average
    of all those autocorrelations.
    """
    corr = np.array([
        np.correlate(mat[:,i] - mat[:,i].mean(), mat[:,i] - mat[:,i].mean(), 'same') / mat[:,i].var()
        for i in xrange(mat.shape[1])
    ]).mean(axis=0)
    corr = corr[len(corr)/2:]  # Take only the right side
    
    return corr / np.arange(mat.shape[0], mat.shape[0] - len(corr), -1)


def weights_from_corr_1d(corr, size):
    corr = corr.copy()
    corr[corr < 0] = 0
    W = scipy.sparse.diags(np.tile(corr[1:], (size-1, 1)).T, np.arange(1, len(corr)))
    W = W + W.T
    return W


def weights_to_lap(W):
    return scipy.sparse.diags(W.sum(axis=0).flat) - W


def time_correlation_laplacian(mat):
    corr = autocorrelation(mat)
    corr /= corr.max()
    corr = corr[:np.argwhere(corr <= 0)[0,0]]
    return weights_to_lap(weights_from_corr_1d(corr, mat.shape[0]))


def weights_from_corr_2d(corr, shape):
    offsets1 = np.arange((-corr.shape[0]+1)/2, (corr.shape[0]+1)/2)
    offsets2 = offsets1 * shape[0]
    offsets = np.array(offsets1 + np.matrix(offsets2).T).flatten()
    
    size = shape[0] * shape[1]
    W = scipy.sparse.spdiags(np.tile(corr.flat, (size, 1)).T, offsets, size, size)
    W = (W + W.T)/2
    W.setdiag(0)
    
    return W


def corr_list_from_interpol(interp, corrlen, size, angs):
#     for length in range(1, len(angs)/2):
#         if np.abs(angs[length:] - angs[:-length]).min() > corrlen:
#             break
    length = len(angs)  # todo: do something smarter?
    corrs = np.zeros((size, 2*length + 1))
    for t in xrange(size):
        angs_diff = np.abs(angs - angs[t])[max(0, t-length):min(t+length+1, len(angs))]
        
        if corrlen is None:
            left, right = 0, len(angs_diff)
        else:
            left = np.argwhere(angs_diff <= corrlen)[0,0]
            try:
                right = left + np.argwhere(angs_diff[left:] > corrlen)[0,0]
            except IndexError:
                right = len(angs_diff)
        
        corrs[t, max(0, length-t+left):min(length-t+right, 2*length+1)] = np.maximum(0, interp(angs_diff[left:right]))
        
    return corrs


def weights_from_corr_1d_interp(interp, corrlen, size, angs):
    corrs = corr_list_from_interpol(interp, corrlen, size, angs)
    W = scipy.sparse.diags(corrs.T, np.arange(-(corrs.shape[1]/2), corrs.shape[1]/2+1), (size,size))
    W = W + W.T
    W.setdiag(0)
    return W


def temporal_corr_weights(mat, angs, tlen):
    mat_interp, angs_interp = interpolate_resample_cube(mat, angs, step=np.abs(angs[1:] - angs[:-1]).min())
    ac = autocorrelation(mat_interp)
    interp = scipy.interpolate.interp1d(
        np.abs(angs_interp[:len(ac)] - angs_interp[0]),
        ac,
        fill_value=0., bounds_error=False
    )
    
    return weights_from_corr_1d_interp(interp, tlen, mat.shape[0], angs)


def distance_weights(mat, mask, only_mask=False, thresh=.4, sig=1., norm=1):
    """
    If `only_mask` is True, the weights are only computed for columns that
    appear in the mask.  This is useful to reduce computational load.
    """
    if norm == 1:
        std = np.abs(mat[~mask] - mat[~mask].mean()).mean()
    elif norm == 2:
        std = mat[~mask].std()
    else:
        raise ValueError("`norm` should be 1 or 2")
    
    if only_mask:
        # Include only columns that are part of the mask
        columns = np.any(mask, axis=0)[1:]
    else:
        columns = slice(None)
    
    weights = []
    I = []
    J = []
    # Iterate over every column in `mat`, starting from 1
    # (excluding those not in `columns`)
    for j in np.arange(1, mat.shape[1])[columns]:
        
        # `D` is the difference between the columns up to `j`, and `j` itself
        # From this, we exclude `mask[:, :j]` because these elements are masked,
        # and `mask[:, [j]]` because these are masked in the `j`th column
        # They are set to 0 in `D`, and removed from the count below by summing
        # over ~vector_mask
        vector_mask = mask[:, :j] | mask[:, [j]]
        D = (mat[:, :j] - mat[:, [j]])
        D[vector_mask] = 0
        
        # This might have x / 0 divisions (=> +Inf or NaN), but it's not a problem
        with np.errstate(divide='ignore', invalid='ignore'):
            if norm == 1:
                d = np.linalg.norm(D, ord=1, axis=0) / np.sum(~vector_mask, axis=0)
            elif norm == 2:
                d = np.linalg.norm(D, ord=2, axis=0) / np.sqrt(np.sum(~vector_mask, axis=0))
        
        # If d[i] is NaN, d[i] < thresh * d will be False for i, so NaNs are not a problem
        # If d[i] is +Inf, d[i] < thresh * d will be False for i, so +Infs are not a problem
        with np.errstate(invalid='ignore'):
            i = np.argwhere(d < thresh * std).flatten()
        
        if norm == 1:
            w = np.exp(-d[i] / (std * sig))
        elif norm == 2:
            w = np.exp(-(d[i] / (std * sig))**2)
        
        weights += w.tolist()
        I += i.tolist()
        J += [j] * len(i)
    
    W = scipy.sparse.coo_matrix((weights, (I, J)), shape=(mat.shape[1], mat.shape[1]))
    W = W + W.T
    
    return W
