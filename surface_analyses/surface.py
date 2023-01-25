"""
Simple interface for the more complete plyfile package.

Stores vertices and faces, as well as any number of data, which are stored as
properties of the vertices.

Notes
-----
* All properties are written as i4, u4, or f8. This means that e.g. we write
  colors as u4 or i4 rather than u1.
"""
import plyfile
import numpy as np
import matplotlib.cm
import matplotlib.colors

PLY_DTYPES = {
    'i': 'i4',
    'f': 'f8',
    'u': 'u4',
}

class Surface:
    def __init__(self, verts, faces):
        """
        Parameters
        ----------
        verts: np.ndarray, shape=(n_verts, 3)
            vertices in Angstrom. Will be converted to nm, when writing a ply file.
        faces: np.ndarray, shape=(n_verts, 3)
            Triangles defined as 3 integer indices for verts.
        """
        self.vertices = np.atleast_2d(verts)
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
            self['red'] = red
        if green is not None:
            self['green'] = green
        if blue is not None:
            self['blue'] = blue

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
        vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        keys = list(self.data.keys())
        for k in keys:
            dtype = np.asarray(self.data[k]).dtype
            if dtype.kind not in PLY_DTYPES:
                raise ValueError(f"dtype={dtype} arrays in data unsupported by the Surface class")
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
            dtype=[('vertex_indices', 'i4', (3,))],
        )
        face = plyfile.PlyElement.describe(face_arr, "face")
        out = plyfile.PlyData([vertex, face], text=text)
        return out

    def write_ply(self, fname):
        with open(fname, mode='wb') as f:
            self.as_plydata().write(f)

    @classmethod
    def from_plydata(cls, plydata):
        pass


def color_surface_by_patch(surf, patches, cmap='tab20c'):
    """Given a list of patches, give each patch a color from the cmap.

    Parameters
    ----------
    surf: Surface
    patches: List[1D integer np.ndarray]
        vertex indices per patch
    cmap: matplotlib colormap
        valid argument for matplotlib.cm.get_cmap
    """
    cmap = matplotlib.cm.get_cmap(cmap)
    values = np.full(surf.n_vertices, len(patches))
    for i, patch in enumerate(patches):
        values[patch] = i
    colors = cmap(values)[:, :3] * 256
    not_in_patch = values == len(patches)
    colors[not_in_patch] = 256
    surf.set_color(*colors.T)


def color_surface(surf, data, cmap='coolwarm_r', clip_fraction=0.1, clim=None):
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
        self._halfrange = max(self._vcenter-qlower,
                              qhigher-self._vcenter)
        self._set_vmin_vmax()
