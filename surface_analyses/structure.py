from itertools import chain
from prmtop.crippen import aa_sasa
import mdtraj as md
import numpy as np
import sap.sap as sap
import TMalign_wrapper.io as tm


def load_aligned_trajectory(filenames, topname, stride, ref, sel):
    traj = load_trajectory(filenames, topname, stride, sel)
    if ref is not None:
        print('Aligning trajectory')
        traj = align_trajectory(traj, ref)
    return traj


def load_trajectory(filenames, topname, stride, sel):
    top = md.load_topology(topname)
    atoms = top.select(sel)
    stride_dict = {} if stride == 1 else {'stride': stride}
    return md.join([
        md.load(t, top=top, **stride_dict, atom_indices=atoms)
        for t in filenames
    ])


def align_trajectory(traj, reference: str):
    ref = md.load(reference)
    if any(r.name == 'HOH' for r in chain(traj.top.residues, ref.top.residues)):
        raise RuntimeError("water is left in the topologies")
    alignment = tm.MDTrajSequenceAlignment.from_trajs(traj[0], ref, all_chains=True)
    return alignment.align(traj)
    # return md.join(
    #     tm.Alignment.from_mdtraj(frame, ref).transform_mdtraj(frame)
    #     for frame in traj
    # )


def heavy_atom_grouper(atoms):
    prev_heavy_atom = list(prev_heavy(atoms))
    def group(x):
        return sum_by(prev_heavy_atom, x, 0)
    return group


def prev_heavy(atoms):
    heavy = -1
    for at in atoms:
        if at.atomic_number != 1:
            heavy += 1
        if heavy == -1:
            raise ValueError('First atom must be heavy')
        yield heavy


def sum_by(assignment, numbers, minlength):
    return np.bincount(assignment, numbers, minlength)
sum_by = np.vectorize(sum_by, signature='(n),(n),()->(m)')


def saa_ref(traj, atoms, surftype):
    if surftype == 'normal':
        ref = 1.
    elif surftype == 'atom_norm':
        ref = np.array([
            get_ref_surf(a.residue_label, a.name)
            for a in atoms
        ])
    elif surftype == 'sc_norm':
        ref = sidechain_saa_ref(traj)
    else:
        raise ValueError('Unknown surftype: ' + str(surftype))
    return ref


def sidechain_saa_ref(traj):
    saa = sidechain_saa_per_atom(traj.top)
    bb_sel = traj.top.select('backbone or (name H) or (name HA) or (name HA2)')
    saa[bb_sel] = np.inf
    return saa


def sidechain_saa_per_atom(top, sidechain_saa=None):
    if sidechain_saa is None:
        sidechain_saa = sap._default_sidechain_saa
    return np.array([
        sidechain_saa[at.residue.name]
        for at in top.atoms
    ])


def get_ref_surf(residue, atom):
    key = (residue, atom)
    from_table = aa_sasa().get(key, np.nan)
    if np.isnan(from_table) and atom in {'H1', 'H2', 'H3'}:
        # using LYS as a reference for the N terminus
        n_term_h = aa_sasa().get(('LYS', 'HZ1'), np.nan)
        return n_term_h
    elif np.isnan(from_table) and atom == 'OXT':
        o_term = aa_sasa().get(('ASP', 'OD1'), np.nan)
        return o_term
    return from_table
