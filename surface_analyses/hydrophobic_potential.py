# from collections import namedtuple
import math
import warnings

from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates
import gisttools as gt
from skimage.measure import marching_cubes
from skimage.filters import gaussian
import numpy as np

import gisttools as gt

from .surface import Surface

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
    radii = np.array([a.element.radius for a in traj.top.atoms])
    surfaces = []
    for frame in traj.xyz:
        pot = MoeHydrophobicPotential(frame, propensities, rcut, alpha)
        extent = cubic_extent(frame) + 2*rmax
        grid = cubic_grid_with_extent(extent, spacing)
        solv_dist = ses_grid(grid, frame, radii, solvent_radius=solv_rad, rmax=rmax)
        verts, faces, _, _ = solvent_surface(solv_dist, grid)
        surf_vals = pot.evaluate(verts)
        tree = cKDTree(frame)
        _, closest_atom = tree.query(verts)
        blurred = gaussian_grid(grid, frame, blur_sigma)
        grid_coords = (verts - grid.origin) / grid.delta
        blurred_lvl = map_coordinates(blurred, grid_coords.T, order=1)
        surfaces.append(Surface(verts, faces))
        surfaces[-1]['values'] = surf_vals
        surfaces[-1]['atom'] = closest_atom
        surfaces[-1]['blurred_lvl'] = blurred_lvl
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

def cubic_extent(xyz):
    """Return HALF the side length of a cube centered at the origin that 
    fits all coordinates in xyz."""
    return np.max(np.concatenate([xyz, -xyz]))


def cubic_grid_with_extent(extent, spacing):
    TINY = 1e-10
    min_vox = round((extent+TINY) / spacing) * spacing
    n_vox = int(min_vox / spacing) * 2 + 1
    return gt.grid.Grid(-min_vox, n_vox, spacing)


def ses_grid(grid, xyz, radii, rmax=.3, solvent_radius=.14, fill_val=1000.):
    """Distance to a sphere of radius solvent_radius rolling over the surface 
    defined by xyz and radii.
    
    Returns fill_val where the distance is > rmax.
    Returns -fill_val where the sphere is not excluded.
    """
    distbuffer = sas_grid(grid, xyz, radii, rmax=rmax, fill_val=fill_val)
    verts, _, _, _ = marching_cubes(
        distbuffer.reshape(grid.shape),
        level=solvent_radius,
        spacing=tuple(grid.delta)
    )
    verts += grid.origin
    outside = distbuffer > solvent_radius
    ind, _, solvent_dist = grid.distance_to_centers(verts, rmax)
    solvent_dist -= solvent_radius
    distbuffer[~outside] = fill_val
    distbuffer[ind] = solvent_dist
    distbuffer[outside] = -fill_val
    return distbuffer


def sas_grid(grid, xyz, radii, rmax=.3, fill_val=1000.):
    """Distance to atoms (xyz), or, when the distance is > rmax, fill_val.
    
    Returns
    -------
    fulldist: np.array, shape=(grid.size,)
        For each voxel closer to xyz than rmax, contains the distance to 
        the closest point in xyz. For other voxels, contains fill_val.
    """
    ind, closest, dist = grid.distance_to_spheres(xyz, rmax, radii)
    fulldist = np.full(grid.size, fill_val, dtype='float32')
    fulldist[ind] = dist
    return fulldist


def gaussian_grid(grid, xyz, sigma):
    i_vox = grid.assign(xyz)
    outside = i_vox == -1
    if np.any(outside):
        warnings.warn("Atoms are outside the grid for gaussian_grid")
        i_vox = i_vox[~outside]
    out = np.float64(np.bincount(i_vox, minlength=grid.size)).reshape(grid.shape)
    return gaussian(out, sigma=sigma/grid.delta)

def gaussian_grid_variable_sigma(grid, xyz, sigma, rmax=None):
    """sum(exp(-dist(x, x_i)**2)/(2*sigma_i**2)) at each grid point x, where
    the sum is over every x_i out of xyz.

    Parameters
    ----------
    grid : gisttools.grid.Grid
    xyz : np.ndarray, shape (n_atoms, 3)
    sigma : np.ndarray, shape (n_atoms,)
        sigma_i for every atom (related to the atomic radius).
    rmax : np.ndarray, shape (n_atoms,)
        distance cutoff per atom. Default: 3*sigma
    """
    xyz = np.atleast_2d(xyz)
    assert len(xyz.shape) == 2 and xyz.shape[1] == 3
    n_atoms = xyz.shape[0]
    sigma = np.asarray(sigma)
    assert sigma.shape == (n_atoms,)
    if rmax is None:
        rmax = 3 * sigma
    rmax = np.asarray(rmax)
    assert rmax.shape == (n_atoms,)

    out = np.zeros(grid.size, float)
    for xyz_i, sigma_i, rmax_i in zip(xyz, sigma, rmax):
        ind, dist = grid.surrounding_sphere(xyz_i, rmax_i)
        out[ind] += np.exp(-dist**2/(2*sigma_i**2))
    return out.reshape(grid.shape)

def solvent_surface(solvdist, grid):
    verts, faces, normals, values = marching_cubes(
        solvdist.reshape(grid.shape),
        level=0,
        spacing=tuple(grid.delta),
        gradient_direction='ascent',
        allow_degenerate=False
    )
    verts += grid.origin
    return verts, faces, normals, values


def distance(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.sqrt(np.sum((a-b)**2, axis=-1))
