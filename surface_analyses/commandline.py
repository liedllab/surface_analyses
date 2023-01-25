import datetime
import logging
import sys
import warnings

import mdtraj as md
import numpy as np

from .hydrophobic_potential import hydrophobic_potential
from .structure import load_aligned_trajectory, heavy_atom_grouper, saa_ref
from .propensities import get_propensity_mapping
from .prmtop import RawTopology
from .pdb import PdbAtom
from .sap import blur as sap_blur
from .patches import find_patches, triangles_area
from .surface import color_surface_by_patch, color_surface

def main(args=None):
    print(f"surfscore starting at {datetime.datetime.now()}")
    if args is None:
        args = sys.argv[1:]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('parm')
    parser.add_argument('trajs', nargs='+')
    parser.add_argument('--ref', default=None)
    parser.add_argument('--scale', required=True, help='Hydrophobicity scale in table format, or "crippen" or "eisenberg", or "rdkit-crippen".')
    parser.add_argument('--smiles', type=str, help='SMILES for rdkit-crippen. Use e.g. @smiles.txt to read them from a file.')
    parser.add_argument('--out', type=argparse.FileType('wb'), required=True, help='Output in .npz format')
    parser.add_argument('--stride', default=1, type=int)
    parser.add_argument('--surftype', choices=('normal', 'sc_norm', 'atom_norm'), default='normal')
    parser.add_argument('--group_heavy', action='store_true')

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
    pot_parser = parser.add_argument_group('Hydrophobic potential options')
    pot_parser.add_argument('--potential', action='store_true')
    pot_parser.add_argument('--rmax', default=0.3, type=float)
    pot_parser.add_argument('--solv_rad', default=0.14, type=float)
    pot_parser.add_argument('--grid_spacing', help='Grid spacing in NANOMETERS [nm]', default=.05, type=float)
    pot_parser.add_argument('--rcut', help='rcut parameter for Heiden weighting function [nm]', default=.5, type=float)
    pot_parser.add_argument('--alpha', help='alpha parameter for Heiden weighting function [nm^-1]', default=15., type=float)
    pot_parser.add_argument('--blur_sigma', help='Sigma for distance to gaussian surface [nm]', default=.6, type=float)
    pot_parser.add_argument('--ply_out', help='Output .ply file of first frame for PyMOL')
    pot_parser.add_argument('--patches', action='store_true', help='Output patches instead of hydrophobic potential')
    pot_parser.add_argument('--patch_min', type=float, default=0.12, help='Minimum vertex value to count as a patch')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args(args)

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    print('Loading Trajectory')
    traj = load_aligned_trajectory(
        args.trajs,
        args.parm,
        args.stride,
        ref=args.ref,
        sel='not resname HOH'
    )
    logging.info(f"Loaded trajectory: {traj}")
    atoms = get_atoms_list(args.parm)
    strip_h = args.scale == 'eisenberg' and not args.group_heavy
    if strip_h:
        logging.info("Stripping hydrogen atoms")
        atoms = [a for a in atoms if is_heavy(a)]
        traj = traj.atom_slice(traj.top.select('not element H'))
    print(f'Loaded {traj.n_frames} frames with {traj.n_atoms} atoms.')
    if args.group_heavy:
        grouper = heavy_atom_grouper(atoms)
        coords = traj.atom_slice(traj.top.select('not element H')).xyz
    else:
        grouper = lambda x: x
        coords = traj.xyz
    if args.scale == 'rdkit-crippen':
        if args.smiles is None:
            raise ValueError("--smiles is needed with Scale 'rdkit-crippen'")
        propensities = rdkit_crippen_logp(args.parm, args.smiles)
    else:
        propensity_map = get_propensity_mapping(args.scale)
        propensities = np.asarray(grouper([propensity_map(a) for a in atoms]))
        if args.smiles is not None:
            warnings.warn("--smiles specified but not used.")
    output = {}
    output['propensities'] = propensities
    if any((args.sap, args.surfscore)):
        print('Computing SAA')
        saa_unref = grouper(md.shrake_rupley(traj))
        ref = grouper(saa_ref(traj, atoms, args.surftype))
        output['saa_unref'] = saa_unref
        saa = saa_unref / ref
        output['saa'] = saa
    if args.surfscore:
        print('Computing Surface score')
        output['surfscore'] = saa * propensities[np.newaxis, :]
    if args.sh:
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
            sh_ca = sap_blur(frame, propensities[ca[has_ca]], args.sh_rad)
            # remove self-interaction in the surrounding hydrophobicity
            sh_ca -= propensities_ca
            sh_per_residue = np.zeros(n_res, dtype=float)
            sh_per_residue[has_ca] = sh_ca
            sh_per_atom.append(sh_per_residue[i_res])
        output['surrounding_hydrophobicity'] = np.array(sh_per_atom)
    if args.sap:
        print('Applying SAP blur')
        output['sap'] = np.array([
            sap_blur(frame, frame_saa * propensities, args.blur_rad)
            for frame, frame_saa in zip(coords, saa)
        ])
    if args.potential:
        print('Applying Hydrophobic potential')
        surfs = hydrophobic_potential(
            traj,
            propensities,
            rmax=args.rmax,
            spacing=args.grid_spacing,
            solv_rad=args.solv_rad,
            rcut=args.rcut,
            alpha=args.alpha,
            blur_sigma=args.blur_sigma,
        )
        lists = {
            'verts': [s.vertices for s in surfs],
            'faces': [s.faces for s in surfs],
            'values': [s['values'] for s in surfs],
            'atom': [s['atom'] for s in surfs],
            'blurred_lvl': [s['blurred_lvl'] for s in surfs],
        }
        output['hydrophobic_potential'] = lists
        if args.patches:
            print(f'Starting patch output, patch_min={args.patch_min}')
            patches = []
            print('i_frame,i_patch,patch_size[nm^2]')
            for i_frame, surf in enumerate(surfs):
                area = triangles_area(surf.vertices[surf.faces])
                pat = find_patches(surf.faces, surf['values'] > args.patch_min)
                patches.append(pat)
                for ip, p in enumerate(pat):
                    size = area[p].sum()
                    print(f"{i_frame},{ip},{size}")
        if args.ply_out:
            surf0 = surfs[0]
            if args.patches:
                color_surface_by_patch(surf0, patches[0])
            else:
                color_surface(surf0, 'values')
            surf0.write_ply(args.ply_out)
    np.savez(args.out, **output)
    args.out.close()

def rdkit_crippen_logp(pdb, smiles):
    import rdkit.Chem.AllChem
    import rdkit.Chem.Crippen
    mol = rdkit.Chem.MolFromPDBFile(pdb, removeHs=False)
    ref = rdkit.Chem.MolFromSmiles(smiles)
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
    # In mdtraj, the hydrogens of the N-terminal residue are not bonded to
    # anything. Bond them to the respective N atom.
    # no_bonds = [a for a in atoms if len(a.bonded_atoms) == 0]
    # for a in no_bonds:
    #     resid = a.residue_id
    #     if a.atomic_number != 1:
    #         warnings.warn(f"Encountered a heavy atom without bonds: {a}. Cannot form bonds for this atom.")
    #         continue
    #     # Expect that the N is before the H, but in the same residue, and
    #     # within the first 20 atoms.
    #     for other in atoms[:20][:a.i]:
    #         if other.residue_id == resid and other.name == 'N':
    #             a._bond(other)
    #             break
    #     else:
    #         warnings.warn(f'Could not bond atom {a}')
    # no_bonds = [a for a in atoms if len(a.bonded_atoms) == 0]
    # if no_bonds:
    #     raise ValueError(f"The following atoms have no bond partners: {no_bonds}")
    return atoms

def is_heavy(atom):
    return atom.atomic_number != 1
