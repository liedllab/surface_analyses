#!/usr/bin/env python3

from datetime import datetime
import os
import pathlib
import pprint
import sys
import subprocess

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from gisttools.gist import load_dx
from mdtraj.core.element import carbon, nitrogen, oxygen, sulfur

from .patches import find_patches, triangles_area
from .surface import Surface
from .surface import color_surface, color_surface_by_patch
from .surface import compute_sas, compute_ses, compute_gauss_surf
from .anarci_wrapper.annotation import Annotation


element_radii = {
    carbon: 1.8,
    nitrogen: 1.5,
    oxygen: 1.3,
    sulfur: 1.8,
}

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('pdb', type=str)
    parser.add_argument('dx', type=str, default=None, nargs='?', help="Optional dx file with the electrostatic potential. If this is omitted, you must specify --apbs_dir")
    parser.add_argument('--apbs_dir', help="Directory in which intermediate files are stored when running APBS. Will be created if it does not exist.", type=str, default=None)
    parser.add_argument('--probe_radius', type=float, help='probe radius in Angstrom', default=1.4)
    parser.add_argument('-o', '--out', default=sys.stdout, type=str, help='Output csv file.')
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
        '--surface_type',
        type=str,
        choices=('sas', 'ses', 'gauss'),
        default='sas',
        help='Which type of molecular surface to produce.'
    )
    parser.add_argument(
        '--ply_out',
        type=str,
        help='Base name for .ply output for PyMOL. Will write BASE-pos.ply and BASE-neg.ply.',
    )
    parser.add_argument(
        '--patch_cmap',
        type=str,
        default='tab20c',
        help='Matplotlib colormap for .ply patches output.',
    )
    parser.add_argument(
        '--ply_cmap',
        type=str,
        default='coolwarm_r',
        help='Matplotlib colormap for .ply potential output.',
    )
    parser.add_argument(
        '--ply_clim',
        type=str,
        default=None,
        help='Colorscale limits for .ply output.',
        nargs=2,
    )
    parser.add_argument(
        '--check_cdrs',
        action='store_true',
        help='For an antibody Fv region as input: check whether patches belong to CDRs.',
    )
    parser.add_argument('--gauss_shift', type=float, default=0.1)
    parser.add_argument('--gauss_scale', type=float, default=1.0)
    args = parser.parse_args(argv)

    print(f'ele_patches.py, {datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}\n')
    print('Command line arguments:')
    print(' '.join(sys.argv))

    if args.dx is None and args.apbs_dir is None:
        raise ValueError("Either DX or APBS_DIR must be specified.")

    if args.dx is not None and args.apbs_dir is not None:
        print("Warning: both DX and APBS_DIR are specified. Will not run APBS "
              "and use the dx file instead.")

    if args.dx is not None:
        dxfile = args.dx
    else:
        run_dir = pathlib.Path(args.apbs_dir)
        if not run_dir.is_dir():
            run_dir.mkdir()
        link_target = os.path.relpath(args.pdb, run_dir.resolve())
        linked_pdb = "input.pdb"
        link_position = run_dir / linked_pdb
        if link_position.exists():
            link_position.unlink()
        link_position.symlink_to(link_target)
        pdb2pqr = run_pdb2pqr(linked_pdb, cwd=run_dir)
        if pdb2pqr.returncode != 0:
            print("Error: pdb2pqr failed:")
            print("pdb2pqr stdout:")
            print(pdb2pqr.stdout)
            print("pdb2pqr stderr:")
            print(pdb2pqr.stderr)
            raise RuntimeError("pdb2pqr failed")
        add_ions_to_apbs_input(run_dir / "apbs.in")
        apbs = run_apbs("apbs.in", cwd=run_dir)
        if apbs.returncode != 0:
            print("Error: apbs failed")
            print("apbs stdout:")
            print(apbs.stdout)
            print("apbs stderr:")
            print(apbs.stderr)
            raise RuntimeError("apbs failed")
        dxfile = str(run_dir / "apbs.pqr-PE0.dx")
    gist = load_dx(dxfile, struct=args.pdb, strip_H=False, colname='DX')
    gist['E_dens'] = gist['DX']
    
    pdb = gist.struct
    # *10 because mdtraj calculates stuff in nanometers instead of Angstrom.
    radii = 10. * np.array([atom.element.radius for atom in pdb.top.atoms])
    columns = ['E_dens']

    print('Run info:')
    pprint.pprint({
        '#Atoms': pdb.xyz.shape[1],
        'Grid dimensions': gist.grid.shape,
        **vars(args),
    })

    print('Calculating triangulated SASA')

    if args.surface_type == 'sas':
        surf = compute_sas(gist.grid, gist.coord, radii, args.probe_radius)
    elif args.surface_type == 'gauss':
        surf = compute_gauss_surf(gist.grid, gist.coord, radii, args.gauss_shift, args.gauss_scale)
    elif args.surface_type == 'ses':
        surf = compute_ses(gist.grid, gist.coord, radii, args.probe_radius)
    else:
        raise ValueError("Unknown surface type: " + str(args.surface_type))

    if args.check_cdrs:
        cdrs = Annotation.from_traj(pdb, scheme='chothia').cdr_indices()
        cdrs = set(cdrs)
        cdr_atoms = set(a.index for a in pdb.top.atoms if a.residue.index in cdrs)
    else:
        cdr_atoms = set()

    pdbtree = cKDTree(pdb.xyz[0] * 10.)

    # The patch searching
    print('Finding patches')
    values = gist.interpolate(columns, surf.vertices)[columns[0]]
    pos_patches = find_patches(surf.faces, values > args.patch_cutoff[0])
    pos_patches = sorted(pos_patches, key=len, reverse=True)

    neg_patches = find_patches(surf.faces, values < args.patch_cutoff[1])
    neg_patches = sorted(neg_patches, key=len, reverse=True)

    # Calculate the area of each triangle, and split evenly among the vertices.
    tri_areas = triangles_area(surf.vertices[surf.faces])
    vert_areas = np.zeros(surf.vertices.shape[0])
    for face, area in zip(surf.faces, tri_areas):
        vert_areas[face] += area/3

    # Put the patches into a DataFrame
    patches = []
    for patch in pos_patches:
        patches.append({
            'type': 'positive',
            'npoints': len(patch),
            'area': np.sum(vert_areas[patch]),
            'cutoff': args.patch_cutoff[0],
            'cdr': check_cdr_patch(pdbtree, cdr_atoms, surf.vertices[patch]),
        })
    for patch in neg_patches:
        patches.append({
            'type': 'negative',
            'npoints': len(patch),
            'area': np.sum(vert_areas[patch]),
            'cutoff': args.patch_cutoff[1],
            'cdr': check_cdr_patch(pdbtree, cdr_atoms, surf.vertices[patch]),
        })
    patches = pd.DataFrame(patches)
    patches.to_csv(args.out)

    # Compute the total solvent-accessible potential.
    within_range, closest_atom, distance = gist.distance_to_spheres(rmax=10, atomic_radii=radii)
    not_protein = distance > args.probe_radius
    accessible = within_range[not_protein]
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
        pos_surf = Surface(surf.vertices, surf.faces)
        color_surface_by_patch(pos_surf, pos_patches, cmap=args.patch_cmap)
        pos_surf.write_ply(args.ply_out + '-pos.ply')

        neg_surf = Surface(surf.verts, surf.faces)
        color_surface_by_patch(neg_surf, neg_patches, cmap=args.patch_cmap)
        neg_surf.write_ply(args.ply_out + '-neg.ply')

        potential_surf = Surface(surf.verts, surf.faces)
        potential_surf['values'] = values
        color_surface(potential_surf, 'values', cmap=args.ply_cmap, clim=args.ply_clim)
        potential_surf.write_ply(args.ply_out + '-potential.ply')
    return


def run_pdb2pqr(pdbfile, cwd=".", ff="amber", name_base="apbs"):
    if not isinstance(cwd, pathlib.Path):
        cwd = pathlib.Path(cwd)
    process = subprocess.run(
        ["pdb2pqr", f"--ff={ff}", pdbfile, name_base + ".pqr", "--apbs-input"],
        capture_output=True,
        cwd=cwd,
    )
    return process


def run_apbs(inputfile, cwd="."):
    process = subprocess.run(
        ["apbs", inputfile],
        capture_output=True,
        cwd=cwd,
    )
    return process


def add_ions_to_apbs_input(fname):
    with open(fname) as f:
        inp = list(f)
    with open(fname, 'w') as f:
        for line in inp:
            f.write(line)
            if line.strip().startswith('temp'):
                print("    ion charge 1.0 conc 0.1 radius 2.0", file=f)
                print("    ion charge -1.0 conc 0.1 radius 2.0", file=f)


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
