#!/usr/bin/env python3

import numpy as np
import pandas as pd
# import mdtraj as md
# from mdtraj.formats.pdb.pdbstructure import PdbStructure
from scipy.spatial import cKDTree
# import gridData as gd
# import itertools
import pprint
import sys
from datetime import datetime
from skimage.measure import marching_cubes_lewiner
from mygisttools.gist import load_dx
from mdtraj.core.element import carbon, nitrogen, oxygen, sulfur
import chothia.hmmsearch_wrapper as hmm
import os

element_radii = {
    carbon: 1.8,
    nitrogen: 1.5,
    oxygen: 1.3,
    sulfur: 1.8,
}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('pdb', type=str)
    parser.add_argument('dx', type=str)
    parser.add_argument('--probe_radius', type=float, help='probe radius in Angstrom', default=1.4)
    parser.add_argument('-o', '--out', default='-', type=argparse.FileType('w'), help='Output csv file.')
    # parser.add_argument(
    #     '-cdr', '--cdr_residues',
    #     nargs='*',
    #     default=(),
    #     help='List of PDB residue numbers that define the CDRs.'
    # )
    parser.add_argument(
        '-c', '--patch_cutoff',
        type=float,
        nargs=2,
        default=[2, -2],
        help='Cutoff for positive and negative patches.'
    )
    parser.add_argument(
        '-ic', '--integral_cutoff',
        type=float,
        nargs=2,
        default=[0.3, -0.3],
        help='Cutoffs for "high" and "low" integrals.'
    )
    args = parser.parse_args()

    print(f'ele_patches.py, {datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}\n')
    print('Command line arguments:')
    print(' '.join(sys.argv))

    # pdb = md.load(args.pdb)
    gist = load_dx(args.dx, struct=args.pdb, strip_H=False)
    basename = os.path.splitext(os.path.basename(args.dx))[0]
    gist['E_dens'] = gist[basename]
    
    pdb = gist.struct
    radii = np.array([
        atom.element.radius for atom in pdb.top.atoms]
    ) * 10.  # because mdtraj calculates stuff in nanometers instead of Angstrom.
        # element_radii[at.element] for at in pdb.top.atoms
    # ])
    columns = ['E_dens']
    # radii = np.array([
    #     atom.element.radius for atom in pdb.top.atoms]
    # ) * 10.  # because mdtraj calculates stuff in nanometers instead of Angstrom.

    print('Run info:')
    pprint.pprint({
        '#Atoms': pdb.xyz.shape[0],
        'Grid dimensions': gist.grid.shape,
        **vars(args),
    })

    print('Calculating triangulated SASA')
    # ---------------------------------------------------------------------
    # This part of the program creates a triangulated SASA of the pdb file.
    #
    # This is done by making a sparse distance matrix of the pdb and the dx
    # file.  The distance matrix is then used to find the closest atom to each
    # voxel (note that the atom radius is subtracted from the distance, which
    # corresponds to a distance to the surface of the atom).
    #
    # Having the distance of each voxel to the respective nearest atom, the
    # triangulated SASA is calculated simply via a marching cubes algorithm
    # from scikit-image.
    # ---------------------------------------------------------------------

    # edge_centers = [(edge[1:] + edge[:-1])/2 for edge in gist.grid.edges]
    # voxel_centers = list(itertools.product(*edge_centers))  # faster than apbsgrid.centers()
    # voxel_centers = gist.grid.xyz(np.arange(gist.grid.n_voxels))

    # gridtree = cKDTree(voxel_centers)
    # dmat = gridtree.sparse_distance_matrix(pdbtree, np.max(radii) + 1.4)

    # grid_ind, at_ind = dmat.nonzero()
    # grid_at_dist = dmat[grid_ind, at_ind].toarray()[0]  # The indexing returns an array of shape=(1, N).
    # grid_at_dist -= radii[at_ind]

    # distance_df = pd.DataFrame.from_dict({'distance': grid_at_dist, 'atom_index': grid_ind})
    # smallest_df = distance_df.groupby('atom_index').min()

    full_distances = np.full(np.prod(gist.grid.shape), 10000.)  # Arbitrarily high number.
    # full_distances[smallest_df.index] = smallest_df.distance.values
    ind, closest, dist = gist.distance_to_spheres(rmax=5., atomic_radii=radii)
    full_distances[ind] = dist

    # Create a triangulated SASA.
    verts, faces, normals, values = marching_cubes_lewiner(
        full_distances.reshape(gist.grid.shape),
        spacing=gist.grid.delta,
        level=args.probe_radius,
        gradient_direction='ascent',
        allow_degenerate=False,
    )
    verts += gist.grid.origin

    cdrs = hmm.select_cdrs_from_trajectory(pdb, definition='chothia')
    cdrs = set(cdrs)
    cdr_atoms = set()
    for i, atom in enumerate(pdb.top.atoms):
        if atom.residue.index in cdrs:
            cdr_atoms.add(atom.index)

    pdbtree = cKDTree(pdb.xyz[0] * 10.)

    # Finally: The patch searching!
    print('Finding patches')
    pos_patches = find_patches_from_cutoff(
        faces=faces,
        values=gist.interpolate(columns, verts)[columns[0]],
        patch_minimum=args.patch_cutoff[0],
    )
    pos_patches = sorted(pos_patches, key=lambda x: len(x), reverse=True)

    neg_patches = find_patches_from_cutoff(
        faces=faces,
        values=gist.interpolate(columns, verts)[columns[0]],
        patch_maximum=args.patch_cutoff[1],
    )
    neg_patches = sorted(neg_patches, key=lambda x: len(x), reverse=True)

    # Calculate the area of each triangle, and split evenly among the vertices.
    tri_areas = triangles_area(verts[faces])
    vert_areas = np.zeros(verts.shape[0])
    for face, area in zip(faces, tri_areas):
        vert_areas[face] += area/3

    # Put the patches into a DataFrame
    patches = []
    for patch in pos_patches:
        patches.append({
            'type': 'positive',
            'npoints': len(patch),
            # 'verts': patch,
            'area': np.sum(vert_areas[patch]),
            'cutoff': args.patch_cutoff[0],
            'cdr': check_cdr_patch(pdbtree, cdr_atoms, verts[patch]),
        })
    for patch in neg_patches:
        patches.append({
            'type': 'negative',
            'npoints': len(patch),
            # 'verts': patch,
            'area': np.sum(vert_areas[patch]),
            'cutoff': args.patch_cutoff[1],
            'cdr': check_cdr_patch(pdbtree, cdr_atoms, verts[patch]),
        })
    patches = pd.DataFrame(patches)
    patches.to_csv(args.out)

    # As a small add-on, compute the total solvent-accessible potential.
    accessible = full_distances > args.probe_radius
    voxel_volume = np.prod(gist.grid.delta)
    accessible_data = gist[columns[0]].values[accessible]
    integral = np.sum(accessible_data) * voxel_volume
    integral_high = np.sum(np.maximum(accessible_data - args.integral_cutoff[0], 0)) * voxel_volume
    integral_pos = np.sum(np.maximum(accessible_data, 0)) * voxel_volume
    integral_neg = np.sum(np.minimum(accessible_data, 0)) * voxel_volume
    integral_low = np.sum(np.minimum(accessible_data - args.integral_cutoff[1], 0)) * voxel_volume
    print('Integrals (total, ++, +, -, --):')
    print(f'{integral} {integral_high} {integral_pos} {integral_neg} {integral_low}')
    return


def boolean_mask(index_array, shape):
    """Create a boolean mask with given shape from an index array."""
    boolean = np.zeros(shape, dtype=bool)
    boolean[index_array] = True
    return boolean


def connected_to(verts, faces, exclude):
    """Find vertices that are directly connected to any vertex in verts via faces.

    Parameters
    ----------
    verts : 1D array or iterable
        Vertices whose neighbors will be searched.
    faces : 2D array, shape = (n, 3)
        Triangles that define connectivity.
    exclude : 1D boolean array
        Vertices where exclude == True will not be reported. Must be longer than the highest number in faces.

    Returns
    -------
    connected_verts : 1D array
        Vertices that are directly connected to verts.

    Examples
    --------
    >>> connected_to(
    ...     [0],
    ...     np.array([[2, 0, 1], [2, 3, 4], [0, 5, 6]]),
    ...     np.array([True, False, True, False, False, False, False])
    ... )
    array([1, 5, 6])
    """
    verts_mask = boolean_mask(verts, exclude.shape)
    relevant_faces = faces[
        verts_mask[faces[:, 0]] | verts_mask[faces[:, 1]] | verts_mask[faces[:, 2]]
    ]
    connected_verts = relevant_faces.flat
    #return np.where(boolean_mask(connected_verts, exclude.shape) & ~exclude)
    # a[b[a]] is equivalent to AND for combining a boolean mask b and an array of indices a.
    return np.unique(connected_verts[~exclude[connected_verts]])


def iterate_connected(start_vertex, n_vertices, faces, filter=None):
    """Iterates connected_to to find all vertices that are directly or indirectly linked to a starting vertex.

    Parameters
    ----------
    start_vertex : int
        Vertex to start the search from
    n_vertices : int
        Number of vertices (Must be higher than np.max(faces)).
    faces : 2D array, shape = (n, 3)
        Triangles that define connectivity, as returned by skimage.measure.marching_cubes_lewiner
    filter : 1D boolean array, shape = (n_vertices)
        Defines which vertices can be used in connections

    Examples
    --------
    >>> faces = np.array([[0, 1, 3], [1, 2, 4], [10, 11, 12]])
    >>> filter = np.ones(13, dtype=bool)
    >>> filter[1] = False
    >>> iterate_connected(0, 13, faces)
    array([0, 1, 2, 3, 4])
    >>> iterate_connected(0, 13, faces, filter=filter)
    array([0, 3])
    """
    is_connected = np.zeros((n_vertices), dtype=bool)
    is_connected[start_vertex] = True
    new = np.array([start_vertex])
    while len(new.flat) > 0:
        new = connected_to(new, faces, is_connected)
        if filter is not None:
            new = new[filter[new]]
        is_connected[new] = True
    return np.nonzero(is_connected)[0]


def find_patches_from_cutoff(faces, values, patch_minimum=None, patch_maximum=None):
    """Find all connected patches that are below patch_minimum or above patch_maximum.

    Parameters
    ----------
    faces : np.ndarray, shape=(n_faces, 3)
        Each row should contain 3 indices that define a triangle. (As returned
        by the marching_cubes_lewiner algorithm in scikit-image)
    values : np.ndarray, shape=(n_vertices,)
        Value of the potential on each vertex.
    patch_minimum : float
        Cutoff for positive patches
    patch_maximum : float
        Cutoff for negative patches. Only ONE of patch_minimum and
        patch_maximum can be given.

    Returns
    -------
    patches : list of np.ndarray objects.
        Each array contains a list of vertex indices that define a patch.
    """
    # ^ is bitwise xor
    if not ((patch_minimum is None) ^ (patch_maximum is None)):
        raise ValueError('Exactly one of patch_minimum and patch_maximum is required.')
    if patch_minimum is not None:
        should_be_in_patch = values > patch_minimum
    else:
        should_be_in_patch = values < patch_maximum
    return find_patches(faces, should_be_in_patch)


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

    n_vertices = should_be_in_patch.shape[0]
    in_patch = np.zeros(n_vertices, dtype=bool)
    patches = []
    while not np.all(in_patch | ~should_be_in_patch):
        first = np.nonzero(~in_patch & should_be_in_patch)[0][0]
        patch = iterate_connected(first, n_vertices, faces, filter=should_be_in_patch)
        in_patch[patch] = True
        patches.append(patch)
    return sorted(patches, key=lambda patch: len(patch), reverse=True)


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


def check_cdr_patch(pdbtree, cdr_atoms, patch_verts):
    """Check whether any atom of the patch is part of the CDRs.

    Parameters
    ----------
    pdbtree : scipy.spatial.cKDTree containing the atom coorinates in Angstrom (!!!)
    cdr_atoms : set of atom indices
    patch_verts : numpy.ndarray, shape=(n_vertices, 3)

    Returns
    -------
    bool : True if any of the vertices belongs to the CDRs

    Notes
    -----
    It is assumed that the radii of all atoms are equal.
    This is of course not true, but usually, only a very
    small portion of vertices will be assigned wrongly.
    """
    _, nearest = pdbtree.query(patch_verts)
    # CDR_atoms = set((at.index for resnum in CDR_def for at in struct.top.residue(resnum).atoms))
    return len(cdr_atoms & set(nearest)) != 0

# def check_cdr_patch(struct, patch_verts, CDR_def):
#     """Check whether any atom of the patch is part of the CDRs.

#     Parameters
#     ----------
#     struct : mdtraj.formats.pdb.pdbstructure.PdbStructure with 1 model
#     patch_verts : numpy.ndarray, shape=(n_vertices, 3)
#     CDR_def : iterable of residue indices

#     Returns
#     -------
#     bool : True if any of the vertices belongs to the CDRs

#     Notes
#     -----
#     It is assumed that the radii of all atoms are equal.
#     This is of course not true, but usually, only a very
#     small portion of vertices will be assigned wrongly.
#     """
#     xyz = np.array(list(struct.iter_positions()))
#     struct_tree = cKDTree(xyz)
#     _, nearest = struct_tree.query(patch_verts)
#     CDR_atoms = set()
#     for i, atom in enumerate(struct.iter_atoms()):
#         if f'{atom.residue_number}{atom.insertion_code}'.strip() in CDR_def:
#             CDR_atoms.add(i)
#     # CDR_atoms = set((at.index for resnum in CDR_def for at in struct.top.residue(resnum).atoms))
#     return len(CDR_atoms & set(nearest)) != 0


if __name__ == '__main__':
    main()
