import numpy as np
import vip_hci as vip
import scipy.optimize
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm


def pixels_in_annulus(inner, outer):
    """Generator for every pixel in an annulus."""
    for x in np.arange(-int(np.ceil(outer)), int(np.ceil(outer))+1):
        for y in np.arange(-int(np.ceil(outer)), int(np.ceil(outer))+1):
            if inner*inner <= x*x + y*y <= outer*outer:
                yield y, x


def cart2rad(y, x):
    """Converts cartesian position (y, x) to radial position (rad, theta)"""
    return np.sqrt(y*y + x*x), np.rad2deg(np.arctan2(y, x))


def rad2cart(rad, theta):
    """Converts radial position (rad, theta) to cartesian position (y, x)"""
    return rad * np.sin(np.deg2rad(theta)), rad * np.cos(np.deg2rad(theta))


def cube_collapse(cube):
    """Collapses a cube into a matrix"""
    return cube.reshape(cube.shape[0], -1)


def cube_expand(mat, filler):
    """Expands a matrix into a cube of shape similar to another"""
    return mat.reshape(filler.shape)


def annulus_mapping(cube, radius, width):
    """Creates a pair `collapse`, `expand` of functions to collapse (reps. expand)
    an annulus of a cube into (resp. from) a matrix."""
    annulus = vip.var.get_annulus(cube[0], radius-width/2, width, output_indices=True)
    
    def collapse(cube):
        return cube[:, annulus[0], annulus[1]]
    
    def expand(mat, filler):
        cube = filler.copy()
        cube[:, annulus[0], annulus[1]] = mat
        return cube
    
    return collapse, expand


def trajectory_pixels(dataset, rad, theta):
    """Compute the pixel locations in a frame of a trajectory.
    
    A trajectory is defined as the path followed by a fixed point in the
    sky of an ADI cube.  For example, companions follow trajectories.
    
    Parameters
    ----------
    angles : np.array
        The angles of the ADI cube, in degrees.
    
    rad : float
    theta : float
        The radial coordinates of the initial position of the trajectory.
    
    Returns
    -------
    ys : np.array
    xs : np.array
        A pair of arrays containing the y and x locations of the trajectory, respectively.
    """
    cy, cx = vip.var.frame_center(dataset.cube)
    
    ys = cy + np.sin(np.deg2rad(-dataset.angles + theta)) * rad
    xs = cx + np.cos(np.deg2rad(-dataset.angles + theta)) * rad
    
    return ys, xs


def trajectory_mask(dataset, rad, theta, prad=1.):
    """Create a mask along a trajectory.
    
    Given a trajectory starting point, creates a boolean 3D array equal to True
    for a disk of radius `prad` * FWHM/2 repeated along that path.
    
    Parameters
    ----------
    dataset : np.array
        The ADI dataset.
    
    rad : float
    theta : float
        The starting point of the trajectory.
    
    prad : float
        The planetary radius, in multiples of FWHM/2.
    
    Returns
    -------
    np.array
        The mask, a 3D boolean array.
    """
    mask = np.zeros_like(dataset.cube, bool)
    pixels = trajectory_pixels(dataset, rad, theta)
    
    radius = prad * dataset.fwhm/2
    r2 = radius*radius
    
    yy, xx = np.ogrid[:dataset.cube.shape[1], :dataset.cube.shape[2]]
    for i, (y, x) in enumerate(zip(*pixels)):
        mask[i] = (xx - x)**2 + (yy - y)**2 <= r2
    
    return mask


def disk_mask(frame, pos, radius):
    y, x = pos
    Y, X = np.ogrid[:frame.shape[0], :frame.shape[1]]
    return (X - x)**2 + (Y - y)**2 <= radius*radius


def interpolate_resample_cube(cube, angles, step=None, frames=None):
    """Creates a new cube with constant angular speed by interpolating an existing cube."""
    if step is None and frames is None:
        raise ValueError("please specify either `step` or `frames`")
    
    if frames is None:
        frames = np.ceil(np.abs(angles[-1] - angles[0])/step) + 1
    
    new_angs = np.linspace(angles[0], angles[-1], frames)
    
    interp = scipy.interpolate.interp1d(angles, cube, axis=0, kind='cubic')
    new_cube = interp(new_angs)
    
    return new_cube, new_angs


def psf_cube(cube, psfn, angles, rad, theta):
    """Creates a zero-filled cube with a copy of the reference PSF along a trajectory."""
    return vip.metrics.cube_inject_companions(
        np.zeros_like(cube), psfn, angles,
        1., 0., rad, theta=theta, verbose=False
    )


###################
### PATH SCORES ###
###################


def flux_loglr_path(res, psfn, angles, rad, theta, mask, norm=1, **kwargs):
    """Computes the flux and log likelihood ratio along a path in a subtracted
    data cube.

    Given a subtracted data cube and a trajectory, finds the flux of a potential
    planet along that trajectory that minimizes the norm of the difference of that
    cube and of an injected planet:

        argmin_a  || (R - a * P) / sigma ||_l

    where `a` is the flux, `R` is the subtracted cube, `P` is the injected planet
    cube, `sigma` is the standard deviation (empirical, computed along time) and
    `l` is the norm.  The l-norm is computed only on the pixels of `mask`.
    
    The log likelihood ratio (`loglr`) is the difference of the log-likelihood of
    the research hypothesis (H1: there is a planet of flux `a`) and of the log-
    likelihood of the null hypothesis (H0: there is no planet, `a=0`).
    
    When `a` is negative, because this value is non-physical, the log likelihood
    ratio is set to zero.

    Parameters
    ----------
    res : np.array, 3d
        The residual cube (original - model PSF).

    psfn : np.array, 2d
        The (normalized) PSF of the telescope.

    angles : np.array
        The parallactic angles of the ADI dataset.
    
    rad : float
    theta : float
        The initial position of the trajectory, in radial coordinates.

    mask : np.array, 3d
        The mask along the trajectory, which was used in the matrix completion that
        produced `cube_h`.

    norm : int, 1 or 2
        The norm used for the likelihood of the flux.

    Returns
    -------
    flux : float
        The flux.
    loglr : float
        The log likelihood ratio.
    """
    P = psf_cube(res, psfn, angles, rad, theta)  # Injected planet

    std = res.std(axis=0)

    # Normalize R and P by the std, and select only the pixels in the mask
    res, P = (res/std)[mask], (P/std)[mask]
    
    if norm == 1:
        # No closed-form solution
        a = scipy.optimize.minimize(lambda a: np.abs(res - a * P).sum(), 0).x[0]
        loglr = np.abs(res).sum() - np.abs(res - a * P).sum()
    
    elif norm == 2:
        # Closed-form solution
        a = np.sum(res * P) / np.sum(P ** 2)
        loglr = (res**2).sum() - ((res - a * P)**2).sum()
    
    else:
        raise ValueError("norm should be 1 or 2")
    
    if a < 0:
        return a, 0
    else:
        return a, loglr
        

def flux_ml_path(*args, **kwargs):
    """See `flux_loglr_path`."""
    return flux_loglr_path(*args, **kwargs)[0]


def std_path(res, mask, **kwargs):
    return res[mask].std()


########################
### FULL CUBE SCORES ###
########################


def flux_cube(res_d, **kwargs):
    return np.median(res_d, axis=0)


def stim_cube(res_d, **kwargs):
    return np.mean(res_d, axis=0) / np.std(res_d, axis=0)


def snr_cube(res_d, fwhm, **kwargs):
    array = np.median(res_d, axis=0)
    
    # return vip.metrics.snr.snrmap(array, fwhm, mode='sss', nproc=1, verbose=False)
    
    # Below is a (modified) copy-paste of the code called by the commented line above.
    # I had to change it because it uses a process pool, which cannot be created from
    # within another process pool.
    # TODO: pull request this
    
    sizey, sizex = array.shape
    snrmap = np.zeros_like(array)
    width = min(sizey, sizex) / 2 - 1.5 * fwhm
    mask = vip.var.get_annulus_segments(array, (fwhm / 2) + 2, width, mode="mask")[0]
    mask = np.ma.make_mask(mask)
    
    yy, xx = np.where(mask)
    coords = zip(xx, yy)

    res = np.array([vip.metrics.snr.snr_ss(array, coord, fwhm, True) for coord in coords])
    yy = res[:, 0]
    xx = res[:, 1]
    snr = res[:, 2]
    snrmap[yy.astype('int'), xx.astype('int')] = snr
    
    return snrmap


def loglr_path_cube(res_d, **kwargs):
    pf = np.maximum(0, np.median(res_d, axis=0))
    sig = np.std(res_d, axis=0)
    
    return np.sum((np.abs(res_d) - np.abs(res_d - pf))/sig, axis=0)
