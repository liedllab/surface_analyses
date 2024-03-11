#!/usr/bin/env python3

import argparse
import datetime
import logging
import sys
import warnings
import os.path

import mdtraj as md
import numpy as np

from .hydrophobic_potential import hydrophobic_potential
from .structure import load_trajectory_using_commandline_args, add_trajectory_options_to_parser, heavy_atom_grouper, saa_ref
from .propensities import get_propensity_mapping
from .prmtop import RawTopology
from .pdb import PdbAtom
from .sap import blur as sap_blur
from .patches import find_patches
from .surface import color_surface_by_patch, color_surface, surfaces_to_dict

def main(args=None):
    print(f"pep_patch_hydrophobic starting at {datetime.datetime.now()}")
    print('Command line arguments:')
    print(' '.join(args or sys.argv))
    args = parse_args(args)
    traj = load_trajectory_using_commandline_args(args)
    # trajectory-related arguments are not passed to run_hydrophobic
    # Note: in contrast to commandline_electrostatic, *parm* IS passed to
    # run_hydrophobic
    parm = args.parm
    del args.parm, args.trajs, args.stride, args.ref, args.protein_ref
    run_hydrophobic(parm, traj, **vars(args))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
    )
    add_trajectory_options_to_parser(parser)
    parser.add_argument('--scale', help=(
        'Hydrophobicity scale in table format, or "crippen" or "eisenberg", '
        'rdkit-crippen", or "file". For rdkit-crippen, parm needs to be in PDB '
        'format, and a SMILES file must be supplied with --smiles. With "file", '
        'a file with pre-assigned values per atom can be supplied via '
        '--atom_propensities (single column, one row per atom).'
    ))
    parser.add_argument('--smiles', type=str, help='SMILES for rdkit-crippen. Use e.g. @smiles.txt to read them from a file.')
    parser.add_argument('--atom_propensities', type=str, help='File with pre-defined atom propensities for "--scale file"')
    parser.add_argument('--out', type=str, help='Output in .npz format')
    parser.add_argument(
        '--surftype',
        help=(
            "Controls the grouping of SASA for surface-area based scores (--surfscore, "
            "--sap, --sh). normal: no normalization is performed. sc_norm: normalization "
            "is performed per side-chain. atom_norm: normalization is performed per atom. "
            "sc_norm and atom_norm assumes that only standard residues occur."
        ),
        choices=('normal', 'sc_norm', 'atom_norm'),
        default='normal',
    )
    parser.add_argument('--group_heavy', action='store_true', help="Assign hydrogen SASA to the previous heavy atom.")

    surfscore_parser = parser.add_argument_group('Surface Score related options')
    surfscore_parser.add_argument('--surfscore', action='store_true')

    sap_parser = parser.add_argument_group('SAP related options')
    sap_parser.add_argument('--sap', action='store_true', help=(
        'Use SAP blur. To reproduce the original algorithm, use the '
        'Black&Mould scale and the --surftype sc_norm option.'
    ))
    sap_parser.add_argument('--blur_rad', type=float, default=0.5, help='Blur radius [nm]')
    sh_parser = parser.add_argument_group('Options for surrounding hydrophobicity')
    sh_parser.add_argument('--sh', action='store_true', help=(
        'Compute surrounding hydrophobicity arrording to '
        'https://www.nature.com/articles/275673a0'
    ))
    sh_parser.add_argument('--sh_rad', type=float, default=0.8, help=(
        'Radius for surrounding hydrophobicity [nm]'
    ))
    pot_parser = parser.add_argument_group(
        'Hydrophobic potential options',
        description=(
            'Compute a hydrophobic potential using the method by Heiden et al. '
            '(J. Comput. Aided Mol. Des. 7, 503–514 (1993)). '
            'A triangulated solvent-excluded surface is created via a marching cubes algorithm '
            'with given grid spacing (--grid_spacing) and solvent radius (--solv_rad), '
            'and the atomic hydrophobicity values from the scale (--scale) are mapped to it '
            'via a sigmoidal distance weighting function with given cutoff (--rcut) and '
            'half height of rcut / 2. The steepness is controlled by --alpha (higher -> steeper). '
            'The potential is output to the npz file. Additionally, a .ply file can be written '
            'for visualization (--ply_out). By default, it contains the potential values, but can also '
            'contain the patches (--patches).'
        )
    )
    pot_parser.add_argument('--potential', action='store_true')
    pot_parser.add_argument('--rmax', default=0.3, type=float)
    pot_parser.add_argument('--solv_rad', default=0.14, type=float)
    pot_parser.add_argument('--grid_spacing', help='Grid spacing for the surface definition in NANOMETERS [nm]', default=.05, type=float)
    pot_parser.add_argument('--rcut', help='rcut parameter for Heiden weighting function [nm]', default=.5, type=float)
    pot_parser.add_argument('--alpha', help='alpha parameter for Heiden weighting function [nm^-1]', default=15., type=float)
    pot_parser.add_argument('--blur_sigma', help='Sigma for distance to gaussian surface [nm]', default=.6, type=float)
    pot_parser.add_argument('--ply_out', help='Output .ply file of first frame for PyMOL')
    pot_parser.add_argument('--ply_cmap', help='Color map for the .ply output')
    pot_parser.add_argument('--ply_clim', nargs=2, help='Colorscale limits for .ply output.')
    pot_parser.add_argument('--patches', action='store_true', help='Output patches instead of hydrophobic potential')
    pot_parser.add_argument('--patch_min', type=float, default=0.12, help='Minimum vertex value to count as a patch')
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser.parse_args(argv)


def run_hydrophobic(
    parm: str,
    traj: md.Trajectory,
    scale: str = None,  # required
    smiles: str = None,
    atom_propensities: str = None,
    out: str = None,
    surftype: str = 'normal',  # normal, sc_norm, atom_norm
    group_heavy: bool = False,
    # surfscore_parser
    surfscore: bool = False,
    # sap_parser
    sap: bool = False,
    blur_rad: float = 0.5,
    # sh_parser
    sh: bool = False,
    sh_rad: float = 0.8,
    # pot_parser
    potential: bool = False,
    rmax: float = 0.3,
    solv_rad: float = 0.14,
    grid_spacing: float = 0.05,
    rcut: float = 0.5,
    alpha: float = 15.,
    blur_sigma: float = 0.6,
    ply_out: str = None,
    ply_cmap: str = None,
    ply_clim: tuple = None,
    patches: bool = False,
    patch_min: float = 0.12,
    verbose: bool = False,
):
    if verbose:
        logging.basicConfig(level=logging.INFO)
    if scale is None:
        raise ValueError("scale is a required argument.")
    atoms = get_atoms_list(parm)
    strip_h = scale == 'eisenberg' and not group_heavy
    if strip_h:
        logging.info("Stripping hydrogen atoms")
        atoms = [a for a in atoms if a.is_heavy]
        traj = traj.atom_slice(traj.top.select('not element H'))
    print(f'Using a trajectory with {traj.n_frames} frames with {traj.n_atoms} atoms.')
    if group_heavy:
        grouper = heavy_atom_grouper(atoms)
        coords = traj.atom_slice(traj.top.select('not element H')).xyz
    else:
        grouper = lambda x: x
        coords = traj.xyz
    if atom_propensities:
        assert scale == "file", "--atom_propensities must be used with --scale file"
    if scale == 'file':
        propensities = np.loadtxt(atom_propensities)
        assert len(propensities) == len(atoms), "Number of atom propensities and atoms don't match!"
    elif scale == 'rdkit-crippen':
        if smiles is None:
            raise ValueError("--smiles is needed with Scale 'rdkit-crippen'")
        propensities = rdkit_crippen_logp(parm, smiles)
    else:
        propensity_map = get_propensity_mapping(scale)
        propensities = np.asarray(grouper([propensity_map(a) for a in atoms]))
        if smiles is not None:
            warnings.warn("--smiles specified but not used.")
    output = {}
    output['propensities'] = propensities
    if any((sap, surfscore)):
        print('Computing SAA')
        saa_unref = grouper(md.shrake_rupley(traj))
        ref = grouper(saa_ref(traj, atoms, surftype))
        output['saa_unref'] = saa_unref
        saa = saa_unref / ref
        output['saa'] = saa
    if surfscore:
        print('Computing Surface score')
        output['surfscore'] = saa * propensities[np.newaxis, :]
    if sh:
        print('Computing Surrounding Hydrophobicity')
        i_res = np.array([a.residue_id for a in atoms])
        print(i_res)
        n_res = np.max(i_res) + 1
        ca = np.full(n_res, -1, dtype=int)
        for i, a in enumerate(atoms):
            if a.name == "CA":
                ca[a.residue_id] = i
        has_ca = ca != -1
        sh_per_atom = []
        propensities_ca = propensities[ca[has_ca]]
        for frame in coords[:, ca[has_ca]]:
            sh_ca = sap_blur(frame, propensities[ca[has_ca]], sh_rad)
            # remove self-interaction in the surrounding hydrophobicity
            sh_ca -= propensities_ca
            sh_per_residue = np.zeros(n_res, dtype=float)
            sh_per_residue[has_ca] = sh_ca
            sh_per_atom.append(sh_per_residue[i_res])
        output['surrounding_hydrophobicity'] = np.array(sh_per_atom)
    if sap:
        print('Applying SAP blur')
        output['sap'] = np.array([
            sap_blur(frame, frame_saa * propensities, blur_rad)
            for frame, frame_saa in zip(coords, saa)
        ])
    if potential:
        print('Applying Hydrophobic potential')
        surfs = hydrophobic_potential(
            traj,
            propensities,
            rmax=rmax,
            spacing=grid_spacing,
            solv_rad=solv_rad,
            rcut=rcut,
            alpha=alpha,
            blur_sigma=blur_sigma,
        )
        output.update(surfaces_to_dict(surfs, basename="hydrophobic_potential"))
        if patches:
            print(f'Starting patch output, patch_min={patch_min}')
            patches = []
            print('i_frame,i_patch,patch_size[nm^2]')
            for i_frame, surf in enumerate(surfs):
                area = surf.areas()
                pat = find_patches(surf.faces, surf['values'] > patch_min)
                patches.append(pat)
                for ip, p in enumerate(pat):
                    size = area[p].sum()
                    print(f"{i_frame},{ip},{size}")
        if ply_out:
            if patches:
                if ply_clim:
                    warnings.warn("--ply_clim is ignored with --patches")
                for surf, patch in zip(surfs, patches):
                    color_surface_by_patch(surf, patch, cmap=ply_cmap)
            else:
                for surf in surfs:
                    color_surface(surf, 'values', cmap=ply_cmap, clim=ply_clim)
            fnames = ply_filenames(ply_out, len(surfs))
            for surf, fname in zip(surfs, fnames):
                    surf.write_ply(fname, coordinate_scaling=10)
    if out is not None:
        np.savez(out, **output)
    return output


def ply_filenames(basename, n) -> list:
    """return [basename] if n==1, otherwise insert indices 1..n before the file
    extension.
    """
    if n == 1:
        return [basename]
    base, ext = os.path.splitext(basename)
    return [
        base + str(i) + ext
        for i in range(n)
    ]


def rdkit_crippen_logp(pdb, smiles):
    import rdkit.Chem.AllChem
    import rdkit.Chem.Crippen
    mol = rdkit.Chem.MolFromPDBFile(pdb, removeHs=False)
    ref = rdkit.Chem.AddHs(rdkit.Chem.MolFromSmiles(smiles))
    mol = rdkit.Chem.AllChem.AssignBondOrdersFromTemplate(ref, mol)
    params = rdkit.Chem.Crippen._GetAtomContribs(mol, force=1)
    return np.array([p[0] for p in params])


def get_atoms_list(fname):
    if fname.endswith('.parm7'):
        return get_parm7_atoms_list(fname)
    elif fname.endswith('.pdb') or fname.endswith('.pdb.gz'):
        return get_pdb_atoms_list(fname)


def get_parm7_atoms_list(fname):
    raw = RawTopology.from_file_name(fname)
    return list(raw.iter_atoms(raw.n_protein_atoms()))


def get_pdb_atoms_list(fname):
    raw = md.load_pdb(fname, standard_names=False)
    raw = raw.atom_slice(raw.top.select('not resname HOH WAT NA CL'))
    atoms = PdbAtom.list_from_md_topology(raw.top)
    return atoms
