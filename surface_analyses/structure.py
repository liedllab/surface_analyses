from itertools import chain
import mdtraj as md
import numpy as np

from .data import AVERAGE_SIDECHAIN_SAA, ATOM_DATA
from .tmalign_wrapper import MDTrajSequenceAlignment

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
    alignment = MDTrajSequenceAlignment.from_trajs(traj[0], ref, all_chains=True)
    return alignment.align(traj)


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
    bb_sel = traj.top.select('backbone or name H HA HA2 or resname ACE NME')
    saa[bb_sel] = np.inf
    return saa


def sidechain_saa_per_atom(top, sidechain_saa=None):
    if sidechain_saa is None:
        sidechain_saa = AVERAGE_SIDECHAIN_SAA
    return np.array([
        sidechain_saa[at.residue.name]
        for at in top.atoms
    ])


def get_ref_surf(residue, atom):
    key = (residue, atom)
    try:
        return ATOM_DATA[key][2]
    except KeyError:
        if atom == "OXT":
            return ATOM_DATA[('ASP', 'OD1')][2]
        elif atom in {'H1', 'H2', 'H3'}:
            return ATOM_DATA[('LYS', 'HZ1')][2]
        else:
            return np.nan
