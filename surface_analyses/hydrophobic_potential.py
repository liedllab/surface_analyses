from collections import namedtuple
import math
import warnings

from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates
import gisttools as gt
from skimage.measure import marching_cubes
from skimage.filters import gaussian
import numpy as np


Surfaces = namedtuple('Surfaces', ['verts', 'faces', 'values', 'atom', 'blurred_lvl'])


def hydrophobic_potential(traj, propensities, rmax, spacing, solv_rad, rcut, alpha, blur_sigma):
    radii = np.array([a.element.radius for a in traj.top.atoms])
    frame_verts = []
    frame_faces = []
    frame_projections = []
    frame_closest_atoms = []
    frame_blurred = []
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
        grid_coords = (frame - grid.origin) / grid.delta
        frame_blurred.append(map_coordinates(blurred, grid_coords.T, order=1))
        frame_verts.append(verts)
        frame_faces.append(faces)
        frame_projections.append(surf_vals)
        frame_closest_atoms.append(closest_atom)
    return Surfaces(frame_verts, frame_faces, frame_projections, frame_closest_atoms, frame_blurred)


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
    fulldist: np.array, shape=(grid.n_voxels,)
        For each voxel closer to xyz than rmax, contains the distance to 
        the closest point in xyz. For other voxels, contains fill_val.
    """
    ind, closest, dist = grid.distance_to_spheres(xyz, rmax, radii)
    fulldist = np.full(grid.n_voxels, fill_val, dtype='float32')
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
