import numpy as np
from itertools import chain
import numba


def find_patches(faces, should_be_in_patch):
    """Find all connected patches from the vertices in should_be_in_patch.

    Parameters
    ----------
    faces : np.ndarray, shape=(n_faces, 3)
        Each row should contain 3 indices that define a triangle. (As returned
        by the marching_cubes_lewiner algorithm in scikit-image)
    should_be_in_patch : np.ndarray, shape=(n_vertices,), dtype=bool
        Defines which vertices should be part of a patch

    Returns
    -------
    patches : list of np.ndarray objects.
        Each array contains a list of vertex indices that define a patch.
    """
    assert len(should_be_in_patch.shape) == 1
    assert len(faces.shape) == 2 and faces.shape[1] == 3
    n_vertices = should_be_in_patch.shape[0]
    not_in_patch = np.ones(n_vertices, dtype=bool)
    patches = []
    n_in_patch = 0
    n_total = len(np.flatnonzero(should_be_in_patch))
    while n_in_patch < n_total:
        first = np.flatnonzero(not_in_patch & should_be_in_patch)[0]
        patch = connected(first, n_vertices, faces, include=should_be_in_patch)
        not_in_patch[patch] = False
        patches.append(patch)
        n_in_patch += len(patch)
    return sorted(patches, key=lambda patch: len(patch), reverse=True)


def find_connected_regions(faces, n_vertices=None):
    """Find all connected patches from the vertices in should_be_in_patch.

    Parameters
    ----------
    faces : np.ndarray, shape=(n_faces, 3)
        Each row should contain 3 indices that define a triangle. (As returned
        by the marching_cubes_lewiner algorithm in scikit-image)
    n_vertices : int
        The total number of vertices that exist in the model. If None, use
        np.max(faces) + 1

    Returns
    -------
    patches : list of np.ndarray objects.
        Each array contains a list of vertex indices that define a patch.
    """
    if n_vertices is None: 
        n_vertices = np.max(faces) + 1
    not_in_patch = np.ones(n_vertices, dtype=bool)
    n_in_patch = 0
    patches = []
    while n_in_patch < len(not_in_patch):
        first = np.flatnonzero(not_in_patch)[0]
        patch = connected(first, n_vertices, faces, include=None)
        not_in_patch[patch] = False
        patches.append(patch)
        n_in_patch += len(patch)
    return sorted(patches, key=lambda patch: len(patch), reverse=True)

def connected(start_vertex, n_vertices, faces, include=None):
    """Find all vertices directly or indirectly linked to a starting vertex.
    
    Parameters
    ----------
    start_vertex : int
        Vertex to start the search from
    n_vertices : int
        Number of vertices (Must be higher than np.max(faces)).
    faces : 2D array, shape = (n, 3)
        Triangles that define connectivity, as returned by
        skimage.measure.marching_cubes_lewiner
    include : 1D boolean array, shape = (n_vertices)
        Defines which vertices can be used in connections. If None, use all
        vertices.

    Examples
    --------
    >>> faces = np.array([[0, 1, 3], [1, 2, 4], [10, 11, 12]])
    >>> include = np.ones(13, dtype=bool)
    >>> include[1] = False
    >>> connected(0, 13, faces)
    array([0, 1, 2, 3, 4])
    >>> connected(0, 13, faces, include=include)
    array([0, 3])
    """
    is_connected = np.zeros((n_vertices), dtype=bool)
    is_connected[start_vertex] = True
    new = np.array([start_vertex])
    while new.size > 0:
        new = directly_connected(new, faces, ~is_connected)
        if include is not None:
            # a[b[a]] is like AND for a boolean mask b and an index array a.
            new = new[include[new]]
        is_connected[new] = True
    return np.flatnonzero(is_connected)


def directly_connected(verts, faces, include):
    """Find vertices directly connected to any vertex in verts via faces.
    
    Parameters
    ----------
    verts : 1D array or iterable
        Vertices whose neighbors will be searched.
    faces : 2D array, shape = (n, 3)
        Triangles that define connectivity.
    include : 1D boolean array
        Only report vertices where include == True. Must be longer than the
        highest number in faces.
    
    Returns
    -------
    connected_verts : 1D array
        Vertices that are directly connected to verts.
        
    Examples
    --------
    >>> directly_connected(
    ...     [0],
    ...     np.array([[2, 0, 1], [2, 3, 4], [0, 5, 6]]),
    ...     np.array([True, False, True, False, False, False, False])
    ... )
    array([1, 5, 6])
    """
    verts_mask = boolean_mask(verts, include.shape)
    connected_verts = connected_to_mask(verts_mask, faces)
    # a[b[a]] is like AND for a boolean mask b and an index array a.
    included_connected = connected_verts[include[connected_verts]]
    verts_mask[:] = False
    verts_mask[included_connected] = True
    return np.flatnonzero(verts_mask)


@numba.njit
def connected_to_mask(verts_mask, faces):
    assert len(faces.shape) == 2
    assert len(verts_mask.shape) == 1
    is_relevant = np.zeros(verts_mask.shape, dtype=np.int8)
    for face in faces:
        p1, p2, p3 = face
        if verts_mask[p1]:
            is_relevant[p2] = True
            is_relevant[p3] = True
        if verts_mask[p2]:
            is_relevant[p1] = True
            is_relevant[p3] = True
        if verts_mask[p3]:
            is_relevant[p2] = True
            is_relevant[p1] = True
    return np.flatnonzero(is_relevant)


def boolean_mask(index_array, shape):
    """Create a boolean mask with given shape from an index array."""
    mask = np.zeros(shape, dtype=bool)
    mask[index_array] = True
    return mask


def triangles_area(triangles):
    """Compute area of each triangle in triangles.
    
    Parameters
    ----------
    triangles : np.ndarray, shape=(n_triangles, 3, 3)
        Eeach triangle must contain 3 rows of xyz coordinates.

    Returns
    -------
    areas : np.ndarray, shape=(n_triangles,)
    
    Examples
    --------
    >>> triangles = [[[0., 0., 0.],
    ...               [0., 0., 3.],
    ...               [0., 4., 0.]],
    ...              [[-1, -1, -1],
    ...               [-1, 0, -1],
    ...               [3, -1, -1]]]
    >>> triangles_area(triangles)
    array([6., 2.])
    >>> triangles_area([[[0, 0, 0], [1, 1, 1]]])
    Traceback (most recent call last):
    AssertionError: shape of triangles must be (n_triangles, 3, 3), not (1, 2, 3).
    """
    triangles = np.asarray(triangles)
    if triangles.size == 0:
        return 0.
    assert len(triangles.shape) == 3 and triangles[0].shape == (3, 3), f"shape of triangles must be (n_triangles, 3, 3), not {triangles.shape}."
    ab = triangles[:, 1, :] - triangles[:, 0, :]
    ac = triangles[:, 2, :] - triangles[:, 0, :]
    cross = np.cross(ab, ac, axis=1)
    cross_abs = np.sqrt(np.sum(cross**2, axis=1))
    return cross_abs / 2.
