import numpy as np


class Empty(object):
    
    def fit(self, dataset):
        self.cube_h = np.zeros_like(dataset.cube)
        return self.model_psf()
    
    def model_psf(self):
        return self.cube_h


class Median(object):
    
    def fit(self, dataset):
        median = np.median(dataset.cube, axis=0)
        self.cube_h = np.repeat(median[np.newaxis, :, :], dataset.cube.shape[0], axis=0)
        return self.model_psf()
    
    def model_psf(self):
        return self.cube_h