import numpy as np
import os, sys
os.environ['MPLBACKEND'] = 'agg'
import vip_hci as vip
import copy
import types

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))


def load_adi_dataset(adi_cube, psf, psf_size=None):
    adi_cube = os.path.join(ROOT_PATH, adi_cube)
    psf = os.path.join(ROOT_PATH, psf)
    dataset = vip.HCIDataset(
        cube=vip.fits.open_fits(adi_cube, n=0),
        angles=vip.fits.open_fits(adi_cube, n=1),
        psf=vip.fits.open_fits(psf)
    )
    
#     # FWHM estimate
#     psf_model = vip.var.fit_2dgaussian(dataset.psf, crop=True, cropsize=9, full_output=True)
#     dataset.fwhm = np.mean([psf_model.fwhm_x, psf_model.fwhm_y])

#     # Normalize PSF
#     dataset.psf = vip.hci_dataset.normalize_psf(dataset.psf, dataset.fwhm, size=21)
#     dataset.psfn = dataset.psf
    
    dataset.normalize_psf(verbose=False, size=psf_size)
    dataset.planets = []
    dataset.px_scale = 0.
    
    return dataset


def bpic_naco():
    dataset = load_adi_dataset(
        "../data/bpic_naco.fits",
        "../data/naco_psf.fits",
        psf_size=21
    )
    dataset.recenter(negative=True, plot=False)
    dataset.planets = [
        (16.26, 45.0)
    ]
    return dataset


def bpic_naco_empty():
    dataset = load_adi_dataset(
        "../data/bpic_naco_empty.fits",
        "../data/naco_psf.fits",
        psf_size=21
    )
    dataset.angles = -dataset.angles
    dataset.recenter(negative=True, plot=False)
    return dataset
    

def sphere_h2_51eri_2016():
    dataset = load_adi_dataset(
        "../data/sphere_h2_51eri_2016.fits",
        "../data/sphere_h2_51eri_2016_psf.fits"
    )
    dataset.recenter(negative=False, plot=False)
    dataset.planets = [
        (30.74, -101.6)  # Deducted from k1 data
    ]
    return dataset
    

def sphere_h2_51eri_2016_empty():
    dataset = sphere_h2_51eri_2016()
    dataset.angles = -dataset.angles
    dataset.planets = []
    return dataset
    

def sphere_k1_51eri_2016():
    dataset = load_adi_dataset(
        "../data/sphere_k1_51eri_2016.fits",
        "../data/sphere_k1_51eri_2016_psf.fits",
        psf_size=21
    )
    dataset.planets = [
        (37.26, -101.6)
    ]
    return dataset
    

def sphere_k1_51eri_2016_empty():
    dataset = sphere_k1_51eri_2016()
    dataset.angles = -dataset.angles
    dataset.planets = []
    return dataset

        
datasets = [
    'bpic_naco',
    'bpic_naco_empty',
    'sphere_h2_51eri_2016',
    'sphere_k1_51eri_2016',
    'sphere_h2_51eri_2016_empty',
    'sphere_k1_51eri_2016_empty',
]

def loader(key):
    if key in datasets:
        return globals()[key]()
    else:
        raise AttributeError('Unknown dataset "{}"'.format(key))

