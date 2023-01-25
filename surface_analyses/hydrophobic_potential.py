# from collections import namedtuple
import math

from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates
import gisttools as gt
from skimage.measure import marching_cubes
import numpy as np

from .surface import Surface, gaussian_grid, compute_ses

# Todo: don't use a cubic grid for hydrophobic_potential. Rectangular is more
# efficient.

def grid_with_walldist(coordinates, walldist, spacing):
    """Return a rectangular gisttools.grid.Grid with given wall distance to the
    coordinates.

    For the purposes of this function, the grid border is defined by the last
    voxel border rather than the voxel centers.
    """
    coordinates = np.atleast_2d(coordinates)
    assert len(coordinates.shape) == 2 and coordinates.shape[1] == 3
    origin = np.min(coordinates, axis=0) - walldist + spacing / 2
    xyzmax = np.max(coordinates, axis=0) + walldist - spacing / 2
    dim = ((xyzmax - origin) // spacing).astype(int) + 1
    return gt.grid.Grid(origin, dim, spacing)


def hydrophobic_potential(traj, propensities, rmax, spacing, solv_rad, rcut, alpha, blur_sigma):
    # calculates in nanometers
    radii = np.array([a.element.radius for a in traj.top.atoms])
    surfaces = []
    for frame in traj.xyz:
        pot = MoeHydrophobicPotential(frame, propensities, rcut, alpha)
        grid = grid_with_walldist(frame, 2*rmax, spacing)
        ses = compute_ses(grid, frame, radii, solv_rad)
        surf_vals = pot.evaluate(ses.vertices)
        tree = cKDTree(frame)
        _, closest_atom = tree.query(ses.vertices)
        blurred = gaussian_grid(grid, frame, blur_sigma)
        grid_coords = (ses.vertices - grid.origin) / grid.delta
        blurred_lvl = map_coordinates(blurred, grid_coords.T, order=1)
        ses['values'] = surf_vals
        ses['atom'] = closest_atom
        ses['blurred_lvl'] = blurred_lvl
        surfaces.append(ses)
    return surfaces


class MoeHydrophobicPotential:
    def __init__(self, centers, logp, rcut, alpha):
        self.centers = centers
        self.logp = logp
        self.rcut = rcut
        self.alpha = alpha
        return
    
    def evaluate_distances(self, distances, nbr_indices):
        nbr_logp = self.logp[nbr_indices]
        wgt = heiden_weight(distances, self.rcut, self.alpha)
        return np.average(nbr_logp, weights=wgt)
    
    def evaluate(self, positions):
        tree = cKDTree(self.centers)
        values = []
        for point, nbrs in zip(positions, tree.query_ball_point(positions, self.rcut)):
            dd = distance(point, self.centers[nbrs])
            values.append(self.evaluate_distances(dd, nbrs))
        return np.array(values)

    
def heiden_weight(x, rcut, alpha):
    """Weighting function by Heiden et al. 1993
    
    Reference:
    Heiden, W., Moeckel, G., Brickmann, J.; J. Comput. Aided Mol. Des. 7 (1993) 503–514.
    
    This is a Fermi function with a scaling factor such that g(0) is 
    always 1, and g(rcut/2) is 0.5. Alpha controls how hard the cutoff 
    is (the higher the steeper)
    
    Parameters:
        x: np.ndarray. Distances to compute weights for.
        rcut: float. Twice the value at which the weight is 0.5
        alpha: float. Controls the steepness of the Fermi function.
    """
    return (
        (math.exp(-alpha * rcut/2) + 1)
        / (np.exp(alpha * (x-rcut/2)) + 1)
    )


def distance(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.sqrt(np.sum((a-b)**2, axis=-1))
