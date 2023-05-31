"""
Simple interface for the more complete plyfile package.

Stores vertices and faces, as well as any number of data, which are stored as
properties of the vertices.

Notes
-----
* All properties are written as i4, u4, or f8. This means that e.g. we write
  colors as u4 or i4 rather than u1.
"""
import warnings

from skimage.filters import gaussian
from gisttools.grid import Grid
from skimage.measure import marching_cubes
import matplotlib.cm
import matplotlib.colors
import numpy as np
import plyfile

PLY_DTYPES = {
    "i": "i4",
    "f": "f8",
    "u": "u4",
}


class Surface:
    def __init__(self, vertices, faces):
        """
        Parameters
        ----------
        vertices: np.ndarray, shape=(n_verts, 3)
            vertices in Angstrom. Will be converted to nm, when writing a ply file.
        faces: np.ndarray, shape=(n_verts, 3)
            Triangles defined as 3 integer indices for vertices.
        """
        self.vertices = np.atleast_2d(vertices)
        self.faces = np.atleast_2d(faces)
        assert len(self.vertices.shape) == 2 and self.vertices.shape[-1] == 3
        assert len(self.faces.shape) == 2 and self.faces.shape[-1] == 3
        self.data = {
            "red": np.full(self.n_vertices, 254.5),
            "green": np.full(self.n_vertices, 100.4),
            "blue": np.full(self.n_vertices, 254.5),
        }

    def __getitem__(self, name):
        return self.data[name]

    def __setitem__(self, name, value):
        value = np.broadcast_to(value, (self.n_vertices,))
        self.data[name] = value

    def set_color(self, red=None, green=None, blue=None):
        if red is not None:
            self["red"] = red
        if green is not None:
            self["green"] = green
        if blue is not None:
            self["blue"] = blue

    @property
    def n_vertices(self):
        return len(self.vertices)

    @property
    def n_faces(self):
        return len(self.faces)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.n_vertices} vertices, {self.n_faces} faces)"

    def as_plydata(self, text=True, units_per_angstrom=0.01):
        """Convert to a plyfile.PlyData object, while scaling coordinates

        the default scaling units_per_angstrom=0.01 matches the PyMol CGO
        scaling factor of 1:100.
        """
        vertex_dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
        keys = list(self.data.keys())
        for k in keys:
            dtype = np.asarray(self.data[k]).dtype
            if dtype.kind not in PLY_DTYPES:
                raise ValueError(
                    f"dtype={dtype} arrays in data unsupported by the Surface class"
                )
            vertex_dtype.append((k, PLY_DTYPES[dtype.kind]))
        vertex_data = []
        for i in range(self.n_vertices):
            vert = list(self.vertices[i] * units_per_angstrom)
            for k in keys:
                vert.append(self.data[k][i])
            vertex_data.append(tuple(vert))
        vertex_arr = np.array(vertex_data, dtype=vertex_dtype)
        vertex = plyfile.PlyElement.describe(vertex_arr, "vertex")
        face_arr = np.array(
            [(list(row),) for row in self.faces],
            dtype=[("vertex_indices", "i4", (3,))],
        )
        face = plyfile.PlyElement.describe(face_arr, "face")
        out = plyfile.PlyData([vertex, face], text=text)
        return out

    def write_ply(self, fname):
        with open(fname, mode="wb") as f:
            self.as_plydata().write(f)

    @classmethod
    def from_plydata(cls, plydata):
        pass

    @classmethod
    def isosurface(cls, grid, values, isovalue, gradient_direction='descent'):
        """Create a Surface object using an isosurface in grid.

        Parameters
        ----------
        grid : Grid
        values : np.ndarray, shape=(grid.size,)
        isovalue : float
        gradient_direction : string, optional
            Gradient direction for skimage.measure.marching_cubes.
            The two options are:
            * descent : Object was greater than exterior
            * ascent : Exterior was greater than object
        """
        vertices, faces, _, _ = marching_cubes(
            values.reshape(grid.shape),
            spacing=grid.delta,
            level=isovalue,
            gradient_direction=gradient_direction,
            allow_degenerate=False,
        )
        vertices += grid.origin
        return cls(vertices, faces)


def compute_sas(
    grid: Grid,
    centers: np.ndarray,
    radii: np.ndarray,
    probe_radius: float,
    fill_val: float = 1e4,
    return_buffer: bool = False,
):
    """Isosurface of the distance to atoms (centers, radii)

    Returns
    -------
    fulldist: np.array, shape=(grid.size,)
        For each voxel closer to xyz than rmax, contains the distance to
        the closest point in xyz. For other voxels, contains fill_val.
    centers: np.ndarray, shape=(n_atoms, 3)
    radii: np.ndarray, shape=(n_atoms,)
    probe_radius: float
    fill_val: float
        fill value to use in the grid. Should be higher than any atomic radius
        + fill_radius.
    """
    full_distances = np.full(grid.size, fill_val, dtype="float32")
    rmax = np.max(radii) + np.max(grid.delta) * 2
    ind, _, dist = grid.distance_to_spheres(centers, rmax=rmax, radii=radii)
    full_distances[ind] = dist
    surf = Surface.isosurface(grid, full_distances, probe_radius, "descent")
    if return_buffer:
        return surf, full_distances
    return surf


def ses_grid(grid, xyz, radii, solvent_radius=0.14, fill_val=1e4):
    """Distance to a sphere of radius solvent_radius rolling over the surface
    defined by xyz and radii.

    Returns fill_val where the distance is > rmax.
    Returns -fill_val where the sphere is not excluded.
    """
    sas, distbuffer = compute_sas(grid, xyz, radii=radii, probe_radius=solvent_radius, return_buffer=True)
    outside = distbuffer > solvent_radius
    rmax = np.max(radii) + np.max(grid.delta) * 2
    ind, _, solvent_dist = grid.distance_to_centers(sas.vertices, rmax)
    solvent_dist -= solvent_radius
    distbuffer[~outside] = fill_val
    distbuffer[ind] = solvent_dist
    distbuffer[outside] = -fill_val
    return distbuffer


def compute_ses(grid, xyz, radii, solvent_radius=0.14):
    solv_dist = ses_grid(grid, xyz, radii, solvent_radius=solvent_radius)
    return Surface.isosurface(grid, solv_dist, 0., 'ascent')


def compute_gauss_surf(
    grid: Grid,
    centers: np.ndarray,
    radii: np.ndarray,
    gauss_shift: float,
    gauss_scale: float,
):
    full_distances = gaussian_grid_variable_sigma(
        grid,
        centers,
        radii * gauss_scale,
    ).ravel()
    return Surface.isosurface(grid, full_distances, gauss_shift, 'ascent')


def gaussian_grid(grid, xyz, sigma):
    i_vox = grid.assign(xyz)
    outside = i_vox == -1
    if np.any(outside):
        warnings.warn("Atoms are outside the grid for gaussian_grid")
        i_vox = i_vox[~outside]
    out = np.float64(np.bincount(i_vox, minlength=grid.size)).reshape(grid.shape)
    return gaussian(out, sigma=sigma / grid.delta)


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
        distance cutoff per atom. Default: 5*sigma
    """
    xyz = np.atleast_2d(xyz)
    assert len(xyz.shape) == 2 and xyz.shape[1] == 3
    n_atoms = xyz.shape[0]
    sigma = np.asarray(sigma)
    assert sigma.shape == (n_atoms,)
    if rmax is None:
        rmax = 5 * sigma
    rmax = np.asarray(rmax)
    assert rmax.shape == (n_atoms,)

    out = np.zeros(grid.size, float)
    for xyz_i, sigma_i, rmax_i in zip(xyz, sigma, rmax):
        ind, dist = grid.surrounding_sphere(xyz_i, rmax_i)
        out[ind] += np.exp(-(dist**2) / (2 * sigma_i**2))
    return out.reshape(grid.shape)


def color_surface_by_patch(surf, patches, cmap="tab20c"):
    """Given a list of patches, give each patch a color from the cmap.

    Parameters
    ----------
    surf: Surface
    patches: List[1D integer np.ndarray]
        vertex indices per patch
    cmap: matplotlib colormap
        valid argument for matplotlib.cm.get_cmap
    """
    values = np.full(surf.n_vertices, -1)
    for i, patch in enumerate(patches):
        values[patch] = i
    color_surface_by_group(surf, values, cmap=cmap)


def color_surface_by_group(surf, group, cmap="tab20c"):
    """Give each group a color from the cmap.

    Parameters
    ----------
    surf: Surface
    group: np.ndarray
        group of each vertex, or negative numbers for no group (white)
    cmap: matplotlib colormap
        valid argument for matplotlib.cm.get_cmap
    """
    cmap = matplotlib.cm.get_cmap(cmap)
    if isinstance(cmap, matplotlib.colors.LinearSegmentedColormap):
        # this converts to float
        group = group / np.max(group)
    colors = cmap(group)[:, :3] * 256
    not_in_patch = group < 0
    colors[not_in_patch] = 256
    surf.set_color(*colors.T)


def color_surface(surf, data, cmap="coolwarm_r", clip_fraction=0.1, clim=None):
    """Given values for each vertex, color surf using the cmap.

    mpl.colors.CenteredNorm is used to scale the values, i.e., 0 will be in the
    middle of the colormap.

    Parameters
    ----------
    surf: Surface
    data: np.ndarray, shape = (n_vertices,)
        vertex values
    cmap: matplotlib colormap
        valid argument for matplotlib.cm.get_cmap
    clip_fraction: float
        fraction (quantile) of the datapoints that should be clipped in the
        color output. If clim is given, this is ignored.
    clim: None or tuple of length 2
        minimum and maximum of the color range.
    """
    if clim is None:
        norm = QuantileSkippingCenteredNorm(clip_fraction=clip_fraction)
    else:
        vmin, vmax = clim
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.cm.get_cmap(cmap)
    values = norm(surf[data])
    colors = cmap(values)[:, :3] * 256
    surf.set_color(*colors.T)


class QuantileSkippingCenteredNorm(matplotlib.colors.CenteredNorm):
    def __init__(self, vcenter=0, halfrange=None, clip=False, clip_fraction=0.1):
        super().__init__(vcenter=vcenter, halfrange=halfrange, clip=clip)
        if clip_fraction >= 1 or clip_fraction < 0:
            raise ValueError("clip_fraction must be >= 0 and < 1")
        self.clip_fraction = clip_fraction

    def autoscale(self, A):
        """
        Scale so that, in one direction, clip_fraction is the fraction of data
        outside of self._halfrange
        """
        A = np.asanyarray(A)
        above = A[A > self._vcenter]
        below = A[A < self._vcenter]
        qlower = np.quantile(below, self.clip_fraction)
        qhigher = np.quantile(above, 1 - self.clip_fraction)
        self.halfrange = abs(max(self.vcenter - qlower, qhigher - self.vcenter))
