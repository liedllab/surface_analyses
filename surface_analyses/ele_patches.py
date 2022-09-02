#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import pprint
import sys
from datetime import datetime
from skimage.measure import marching_cubes
from gisttools.gist import load_dx
from mdtraj.core.element import carbon, nitrogen, oxygen, sulfur
import chothia.hmmsearch_wrapper as hmm
import os
from patches import find_patches, triangles_area
from .surface import Surface
import matplotlib as mpl


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
    parser.add_argument(
        '--ply_out',
        type=str,
        help='Base name for .ply output for PyMOL. Will write BASE-pos.ply and BASE-neg.ply.',
    )
    args = parser.parse_args()

    print(f'ele_patches.py, {datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}\n')
    print('Command line arguments:')
    print(' '.join(sys.argv))

    gist = load_dx(args.dx, struct=args.pdb, strip_H=False)
    basename = os.path.splitext(os.path.basename(args.dx))[0]
    gist['E_dens'] = gist[basename]
    
    pdb = gist.struct
    # *10 because mdtraj calculates stuff in nanometers instead of Angstrom.
    radii = 10. * np.array([atom.element.radius for atom in pdb.top.atoms])
    columns = ['E_dens']

    print('Run info:')
    pprint.pprint({
        '#Atoms': pdb.xyz.shape[0],
        'Grid dimensions': gist.grid.shape,
        **vars(args),
    })

    print('Calculating triangulated SASA')

    full_distances = np.full(gist.grid.size, 10000.)  # Arbitrarily high number.
    ind, closest, dist = gist.distance_to_spheres(rmax=5., atomic_radii=radii)
    full_distances[ind] = dist

    # Create a triangulated SASA.
    verts, faces, normals, values = marching_cubes(
        full_distances.reshape(gist.grid.shape),
        spacing=gist.grid.delta,
        level=args.probe_radius,
        gradient_direction='descent',
        allow_degenerate=False,
    )
    verts += gist.grid.origin

    cdrs = hmm.select_cdrs_from_trajectory(pdb, definition='chothia')
    cdrs = set(cdrs)
    cdr_atoms = set(a.index for a in pdb.top.atoms if a.residue.index in cdrs)

    pdbtree = cKDTree(pdb.xyz[0] * 10.)

    # Finally: The patch searching!
    print('Finding patches')
    values = gist.interpolate(columns, verts)[columns[0]]
    pos_patches = find_patches(faces, values > args.patch_cutoff[0])
    pos_patches = sorted(pos_patches, key=len, reverse=True)

    neg_patches = find_patches(faces, values < args.patch_cutoff[1])
    neg_patches = sorted(neg_patches, key=len, reverse=True)

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
    voxel_volume = gist.grid.voxel_volume
    accessible_data = gist[columns[0]].values[accessible]
    integral = np.sum(accessible_data) * voxel_volume
    integral_high = np.sum(np.maximum(accessible_data - args.integral_cutoff[0], 0)) * voxel_volume
    integral_pos = np.sum(np.maximum(accessible_data, 0)) * voxel_volume
    integral_neg = np.sum(np.minimum(accessible_data, 0)) * voxel_volume
    integral_low = np.sum(np.minimum(accessible_data - args.integral_cutoff[1], 0)) * voxel_volume
    print('Integrals (total, ++, +, -, --):')
    print(f'{integral} {integral_high} {integral_pos} {integral_neg} {integral_low}')

    if args.ply_out:
        pos_surf = Surface(verts, faces)
        color_surface_by_patch(pos_surf, pos_patches)
        pos_surf.write_ply(args.ply_out + '-pos.ply')

        neg_surf = Surface(verts, faces)
        color_surface_by_patch(pos_surf, neg_patches)
        neg_surf.write_ply(args.ply_out + '-neg.ply')
    return

def color_surface_by_patch(surf, patches, cmap='tab20c'):
    cmap = mpl.cm.get_cmap(cmap)
    values = np.full(surf.n_vertices, len(patches))
    for i, patch in enumerate(patches):
        values[patch] = i
    colors = cmap(values)[:, :3] * 256
    not_in_patch = values == len(patches)
    colors[not_in_patch] = 256
    surf.set_color(*colors.T)


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
    return len(cdr_atoms & set(nearest)) != 0


if __name__ == '__main__':
    main()
