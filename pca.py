import numpy as np
# from sklearn.utils.extmath import randomized_svd
from vip_hci.pca import pca


class PCA(object):
    
    def __init__(self, rank=1):
        self.rank = rank
    
    def fit(self, dataset):
        pcs, cube_h, res, res_d, pf = pca(
            dataset.cube, dataset.angles, ncomp=self.rank,
            verbose=False, full_output=True)
        self.cube_h = cube_h
        
        return self.model_psf()
    
    def model_psf(self):
        return self.cube_h


class DummyPCACompletion(object):
    
    def __init__(self, rank=1):
        self.rank = rank
    
    def fit(self, dataset):
        pcs, cube_h, res, res_d, pf = pca(
            dataset.cube, dataset.angles, ncomp=self.rank,
            verbose=False, full_output=True)
        self.cube_h = cube_h
    
    def complete(self, dataset, mask):
        return self.cube_h



# def pca(M, mask=None, rank=1, subtract_mean=False):
#     """Matrix low-rank approximation, encapsulated in the usual "matrix completion" framework.
    
#     THIS IS NOT A MATRIX COMPLETION METHOD.
    
#     Parameters
#     ----------
#     M : np.array
#         The data matrix, a 2D float array.
    
#     mask : np.array
#         The mask of the path in matrix form, a 2D boolean array.
#         THIS IS NOT USED, BECAUSE REMEMBER: THIS IS NOT A MATRIX COMPLETION METHOD.
    
#     rank : int
#         The number of singular values kept in the SVD.
        
#     subtract_mean : bool
#         Whether or not to subtract the mean image before performing PCA.
    
#     Returns
#     -------
#     np.array
#         An approximation of `M` constructed from the first `rank` PCs.
#     """
#     mean = 0
#     if subtract_mean:
#         mean = M.mean(axis=0)
    
#     U, s, VT = randomized_svd(M - mean, n_components=rank, n_iter=5)
#     return np.dot((U * s)[:, :rank], VT[:rank]) + mean
