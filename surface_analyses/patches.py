import numpy as np
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
    faces = np.asarray(faces)
    should_be_in_patch = np.asarray(should_be_in_patch)
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
    return sorted(patches, key=len, reverse=True)

def assign_patches(faces, should_be_in_patch):
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
    patch : np.ndarray, shape=(n_vertices), dtype=int
        The patch number per vertex, or -1 if a vertex is not in a patch.
    """
    assert len(should_be_in_patch.shape) == 1
    assert len(faces.shape) == 2 and faces.shape[1] == 3
    NOT_IN_PATCH = -1
    UNASSIGNED = -2
    n_vertices = should_be_in_patch.shape[0]
    patch = np.full(n_vertices, NOT_IN_PATCH)
    patch[should_be_in_patch] = UNASSIGNED
    n_in_patch = 0
    n_total = len(np.flatnonzero(should_be_in_patch))
    i_patch = 0
    while n_in_patch < n_total:
        first = np.flatnonzero(patch == UNASSIGNED)[0]
        patch_vertices = connected(first, n_vertices, faces, include=should_be_in_patch)
        patch[patch_vertices] = i_patch
        n_in_patch += len(patch_vertices)
        i_patch += 1
    return patch


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
