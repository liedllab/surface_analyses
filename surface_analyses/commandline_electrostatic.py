#!/usr/bin/env python3

import argparse
import csv
from datetime import datetime
from collections import namedtuple
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

from .patches import assign_patches
from .surface import Surface
from .surface import color_surface, color_surface_by_group
from .surface import compute_sas, compute_ses, compute_gauss_surf
from .structure import load_trajectory_using_commandline_args, add_trajectory_options_to_parser

import warnings

element_radii = {
    carbon: 1.8,
    nitrogen: 1.5,
    oxygen: 1.3,
    sulfur: 1.8,
}

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_trajectory_options_to_parser(parser)
    parser.add_argument('--dx', type=str, default=None, nargs='?', help="Optional dx file with the electrostatic potential. If this is omitted, you must specify --apbs_dir")
    parser.add_argument('--apbs_dir', help="Directory in which intermediate files are stored when running APBS. Will be created if it does not exist.", type=str, default=None)
    parser.add_argument('--probe_radius', type=float, help='probe radius in Angstrom', default=1.4)
    parser.add_argument('-o', '--out', default=None, type=str, help='Output csv file.')
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
        '--pos_patch_cmap',
        type=str,
        default='tab20c',
        help='Matplotlib colormap for .ply positive patches output.',
    )
    parser.add_argument(
        '--neg_patch_cmap',
        type=str,
        default='tab20c',
        help='Matplotlib colormap for .ply negative patches output.',
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
    parser.add_argument(
        '-n','--n_patches',
        type=int,
        default=0,
        help='Restrict output to n patches. Positive values output n largest patches, negative n smallest patches.',
    )
    parser.add_argument(
        '-s','--size_cutoff',
        type=float,
        default=0,
        help='Restrict output to patches with an area of over s A^2. If s = 0, no cutoff is applied (default).',
    )
    parser.add_argument('--gauss_shift', type=float, default=0.1)
    parser.add_argument('--gauss_scale', type=float, default=1.0)
    parser.add_argument(
        '--pH',
        type=float,
        default=None,
        help='Specify pH for pdb2pqr calculation. If None, no protonation is performed.',
    )
    parser.add_argument(
        '--ion_species',
        type=str,
        nargs="*",
        default=None,
        help="Specify ion species and their properties (charge, concentration, and radius). "
             "Provide values for multiple ion species as charge1, conc1, radius1, charge2, conc2, radius2, etc."
    )

    args = parser.parse_args(argv)

    print(f'pep_patch_electrostatic, {datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}\n')
    print('Command line arguments:')
    print(' '.join(sys.argv))
    if args.out is None:
        csv_outfile = sys.stdout
    else:
        csv_outfile = open(args.out, "w")

    ion_species = get_ion_species(args)
    traj = load_trajectory_using_commandline_args(args)
    # Renumber residues, takes care of insertion codes in PDB residue numbers
    for i, res in enumerate(traj.top.residues,start=1):
        res.resSeq = i
            
    if args.dx is None and args.apbs_dir is None:
        raise ValueError("Either DX or APBS_DIR must be specified.")

    if args.dx is not None and args.apbs_dir is not None:
        warnings.warn("Warning: both DX and APBS_DIR are specified. Will not run APBS "
              "and use the dx file instead.")

    if traj.n_frames != 1:
        raise ValueError("The electrostatics script only works with a single-frame trajectory.")

    if args.dx is not None:
        griddata = load_dx(args.dx, colname='DX')
        griddata.struct = traj[0]
    else:
        griddata = get_apbs_potential_from_mdtraj(traj, args.apbs_dir, args.pH, ion_species)
    
    # *10 because mdtraj calculates stuff in nanometers instead of Angstrom.
    radii = 10. * np.array([atom.element.radius for atom in traj.top.atoms])
    columns = ['DX']

    print('Run info:')
    pprint.pprint({
        '#Atoms': traj.n_atoms,
        'Grid dimensions': griddata.grid.shape,
        **vars(args),
    })

    print('Calculating triangulated SASA')

    if args.surface_type == 'sas':
        surf = compute_sas(griddata.grid, griddata.coord, radii, args.probe_radius)
    elif args.surface_type == 'gauss':
        surf = compute_gauss_surf(griddata.grid, griddata.coord, radii, args.gauss_shift, args.gauss_scale)
    elif args.surface_type == 'ses':
        surf = compute_ses(griddata.grid, griddata.coord, radii, args.probe_radius)
    else:
        raise ValueError("Unknown surface type: " + str(args.surface_type))

    if args.check_cdrs:
        try:
            from .anarci_wrapper.annotation import Annotation
            cdrs = [
                str(traj.top.residue(i))
                for i in Annotation.from_traj(traj[0], scheme='chothia').cdr_indices()
            ]
        except ImportError as e:
            print(f"CDR annotation failed with the following error:\n{e}\n"
                   "If the error pertains to the annotation tool ANARCI or ANARCI is missing, "
                   "a fresh installation of ANARCI ( https://github.com/oxpig/ANARCI ) or its dependencies might help.\n\n"
                   "To use pep_patch_electrostatic without CDR annotation, rerun the script without the '--check_cdrs' flag." )
            raise RuntimeError("CDR Annotation failed")
    else:
        cdrs = []
    residues = np.array([str(a.residue) for a in traj.top.atoms])

    pdbtree = cKDTree(traj.xyz[0] * 10.)
    _, closest_atom = pdbtree.query(surf.vertices)

    # Calculate the area of each triangle, and split evenly among the vertices.
    tri_areas = surf.areas()
    vert_areas = np.zeros(surf.vertices.shape[0])
    for face, area in zip(surf.faces, tri_areas):
        vert_areas[face] += area/3

    # The patch searching
    print('Finding patches')
    values = griddata.interpolate(columns, surf.vertices)[columns[0]]
    patches = pd.DataFrame({
        'positive': assign_patches(surf.faces, values > args.patch_cutoff[0]),
        'negative': assign_patches(surf.faces, values < args.patch_cutoff[1]),
        'area': vert_areas,
        'atom': closest_atom,
        'residue': residues[closest_atom],
        'cdr': np.isin(residues, cdrs)[closest_atom],
        'value': values,
    })

    # save values and atom in surf for consistency with commandline_hydrophobic
    surf['values'] = values
    surf['atom'] = closest_atom
    
    #keep args.n_patches largest patches (n > 0) or smallest patches (n < 0) or patches with an area over the size cutoff
    if args.n_patches != 0 or args.size_cutoff != 0: 
        #interesting interaction: setting a -n cutoff and size cutoff should yield the n smallest patches with an area over the size cutoff
        replace_vals = {}
        for patch_type in ('negative', 'positive'):
            # sort patches by area and filter top n patches (or bottom n patches for n < 0)
            # also we apply the size cutoff here. It defaults to 0, so should not do anything if not explicitly set as all areas should be > 0.
            area = patches.query(f'{patch_type} != -1').groupby(f'{patch_type}').sum(numeric_only=True)['area']
            order = (area[area > args.size_cutoff] # discard patches with an area under size cutoff
                     .sort_values(ascending=False).index)  # ... and sort them
            
            filtered = order[:args.n_patches] if args.n_patches > 0 else order[args.n_patches:] 
            # set patches not in filtered to -1
            patches.loc[~patches[patch_type].isin(filtered), patch_type] = -1 
            
            # build replacement dict to renumber patches in df according to size
            order_map = {elem : i for i, elem in enumerate(order[:args.n_patches]) }
            replace_vals[patch_type] = order_map        
        patches.replace(replace_vals, inplace=True)
        
    output = csv.writer(csv_outfile)
    output.writerow(['type', 'npoints', 'area', 'cdr', 'main_residue'])
    write_patches(patches, output, 'positive')
    write_patches(patches, output, 'negative')

    # Compute the total solvent-accessible potential.
    within_range, closest_atom, distance = griddata.distance_to_spheres(rmax=10, atomic_radii=radii)
    not_protein = distance > args.probe_radius
    accessible = within_range[not_protein]
    voxel_volume = griddata.grid.voxel_volume
    accessible_data = griddata[columns[0]].values[accessible]
    integral = np.sum(accessible_data) * voxel_volume
    integral_high = np.sum(np.maximum(accessible_data - args.integral_cutoff[0], 0)) * voxel_volume
    integral_pos = np.sum(np.maximum(accessible_data, 0)) * voxel_volume
    integral_neg = np.sum(np.minimum(accessible_data, 0)) * voxel_volume
    integral_low = np.sum(np.minimum(accessible_data - args.integral_cutoff[1], 0)) * voxel_volume
    print('Integrals (total, ++, +, -, --):')
    print(f'{integral} {integral_high} {integral_pos} {integral_neg} {integral_low}')
    
    if args.ply_out:
        pos_surf = Surface(surf.vertices, surf.faces)
        pos_area = patches.query('positive != -1').groupby('positive').sum(numeric_only=True)['area']
        pos_order = pos_area.sort_values(ascending=False).index
        color_surface_by_group(pos_surf, patches['positive'].values, order=pos_order, cmap=args.pos_patch_cmap)
        pos_surf.write_ply(args.ply_out + '-pos.ply')

        neg_surf = Surface(surf.vertices, surf.faces)
        neg_area = patches.query('negative != -1').groupby('negative').sum(numeric_only=True)['area']
        neg_order = neg_area.sort_values(ascending=False).index
        color_surface_by_group(neg_surf, patches['negative'].values, order=neg_order, cmap=args.neg_patch_cmap)
        neg_surf.write_ply(args.ply_out + '-neg.ply')

        potential_surf = Surface(surf.vertices, surf.faces)
        potential_surf['values'] = values
        color_surface(potential_surf, 'values', cmap=args.ply_cmap, clim=args.ply_clim)
        potential_surf.write_ply(args.ply_out + '-potential.ply')

    # close user output file, but not stdout
    if args.out is not None:
        csv_outfile.close()

    return {
        'surface': surf.to_dict(),
        'integrals': {
            'integral': integral,
            'integral_high': integral_high,
            'integral_pos': integral_pos,
            'integral_neg': integral_neg,
            'integral_low': integral_low,
        },
        'patch_vertices': patches.query(f'positive != -1 or negative != -1'),
    }


IonSpecies = namedtuple("IonSpecies", "charge concentration  radius")

DEFAULT_ION_SPECIES = [IonSpecies(1.0, 0.1, 2.0), IonSpecies(-1.0, 0.1, 2.0)]

def get_ion_species(commandline_arguments):
    args = commandline_arguments.ion_species
    if args is None:
        return DEFAULT_ION_SPECIES
    if len(args) % 3 != 0:
        raise ValueError("Number of arguments for --ion_species must be divisible by 3.")
    # important to keep this an iterator
    args_it = (float(arg) for arg in args)
    species = []
    for charge, conc, radius in zip(args_it, args_it, args_it):
        species.append(IonSpecies(charge, conc, radius))
    return species

def get_apbs_potential_from_mdtraj(traj, apbs_dir, pH, ion_species):
    run_dir = pathlib.Path(apbs_dir)
    if not run_dir.is_dir():
        run_dir.mkdir()
    pdb_file = run_dir / "input.pdb"
    traj[0].save_pdb(str(pdb_file), force_overwrite=True)
    pdb2pqr = run_pdb2pqr("input.pdb", cwd=run_dir, pH=pH)
    if pdb2pqr.returncode != 0:
        print("Error: pdb2pqr failed:")
        print("pdb2pqr stdout:")
        print(pdb2pqr.stdout)
        print("pdb2pqr stderr:")
        print(pdb2pqr.stderr)
        raise RuntimeError("pdb2pqr failed")
    add_ions_to_apbs_input(run_dir / "apbs.in", ion_species)
    apbs = run_apbs("apbs.in", cwd=run_dir)
    if apbs.returncode != 0:
        print("Error: apbs failed")
        print("apbs stdout:")
        print(apbs.stdout)
        print("apbs stderr:")
        print(apbs.stderr)
        raise RuntimeError("apbs failed")
    dxfile = str(run_dir / "apbs.pqr-PE0.dx")
    griddata = load_dx(dxfile, colname='DX')
    griddata.struct = traj[0]
    return griddata

def write_patches(df, out, column):
    groups = dict(list(df[df[column] != -1].groupby(column)))
    for patch in sorted(groups.values(), key=lambda df: -df['area'].sum()):
        out.writerow([
            column,
            len(patch),
            patch['area'].sum(),
            np.any(patch['cdr']),
            biggest_residue_contribution(patch),
        ])


def biggest_residue_contribution(df):
    """Find the element in df['residue'] with the highest total contribution in df['area']."""
    return (
        df[['residue', 'area']]
        .groupby('residue')
        .sum()
        ['area']
        .sort_values()
        .index[-1]
    )


def run_pdb2pqr(pdbfile, cwd=".", ff="AMBER", name_base="apbs", pH=None):
    if not isinstance(cwd, pathlib.Path):
        cwd = pathlib.Path(cwd)
    command = ["pdb2pqr", f"--ff={ff}", pdbfile, name_base + ".pqr", "--apbs-input", "apbs.in"]
    if pH is not None:
        command.extend(["--titration-state-method=propka", f"--with-ph={pH}"])
    process = subprocess.run(
        command,
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


def add_ions_to_apbs_input(fname, ion_species):
    with open(fname) as f:
        inp = list(f)
    with open(fname, 'w') as f:
        for line in inp:
            f.write(line)
            if line.strip().startswith('temp'):
                for charge, conc, radius in ion_species:
                    print(f"    ion charge {charge} conc {conc} radius {radius}", file=f)


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
