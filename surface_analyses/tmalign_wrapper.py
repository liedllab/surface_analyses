from subprocess import check_output
from collections import namedtuple
import os.path
from tempfile import TemporaryDirectory
import warnings

import numpy as np


def parseMatrix(filename):
    """Parse a matrix file from TMalign output.

    Parameters
    ----------
    filename : str
        Filename of the matrix output.

    Returns
    -------
    rot : numpy.ndarray, shape=(3,3)
        Rotation matrix.
    trans : numpy.ndarray, shape=(3,)
        Translation vector. Should be applied after rot.

    """
    with open(filename) as f:
        next(f)
        next(f)
        l1 = f.readline().split()
        l2 = f.readline().split()
        l3 = f.readline().split()
    rot = np.array([[float(l1[2]), float(l1[3]), float(l1[4])],
                    [float(l2[2]), float(l2[3]), float(l2[4])],
                    [float(l3[2]), float(l3[3]), float(l3[4])]])
    trans = np.array([float(l1[1]), float(l2[1]), float(l3[1])])
    return rot, trans


class SequenceAlignment(namedtuple('SequenceAlignment', ['residues_a', 'residues_b'])):

    @classmethod
    def from_str(cls, a):
        return cls.from_lines(a.splitlines(keepends=True))

    @classmethod
    def from_lines(cls, lines):
        it = iter(lines)
        for line in it:
            if line.startswith('(":" denotes residue pairs of'):
                a_str = next(it).rstrip()
                next(it)
                b_str = next(it).rstrip()
        return cls(a_str, b_str)

    @staticmethod
    def _get_index(seq1, seq2):
        i = 0
        index = []
        for a_res, b_res in zip(seq1, seq2):
            if a_res != '-' and b_res != '-':
                index.append(i)
            if a_res != '-':
                i += 1
        return index

    def index_a(self):
        return self._get_index(self.residues_a, self.residues_b)

    def index_b(self):
        return self._get_index(self.residues_b, self.residues_a)


class MDTrajSequenceAlignment:
    def __init__(self, atoms, ref):
        self.atoms = atoms
        self.ref = ref

    @classmethod
    def from_trajs(cls, traj, ref, all_chains=False):
        assert traj.n_frames == 1 and ref.n_frames == 1, (
            "MDTrajSequenceAlignment is meant to be instantiated with "
            "single-frame trajectories, although the alignment can be applied "
            "to longer trajectories."
        )
        ca_1 = traj.top.select('name CA')
        ca_2 = ref.top.select('name CA')
        tm_align_output = alignMDTraj(traj, ref, legacy=False, all_chains=all_chains)[2]
        alignment = SequenceAlignment.from_str(tm_align_output.decode('utf-8'))
        atoms = ca_1[alignment.index_a()]
        cropped_ref = ref.atom_slice(ca_2[alignment.index_b()])
        return cls(atoms, cropped_ref)

    def align(self, traj):
        out = traj.slice(slice(None), copy=True)
        out.superpose(
            self.ref,
            atom_indices=self.atoms,
            ref_atom_indices=range(self.ref.n_atoms),
        )
        return out


def runTMalign(fname1, fname2, all_chains=False):
    """Run TMalign to generate an alignment matrix. None of the inputs will be
    overwritten.

    Parameters
    ----------
    fname1 : str
        Filename of 1st PDB file
    fname2 : str
        Filename of 2nd PDB file

    Returns
    -------
    rot : numpy.ndarray, shape=(3,3)
        Rotation matrix.
    trans : numpy.ndarray, shape=(3,)
        Translation vector. Should be applied after rot.

    """
    with TemporaryDirectory() as tmp:
        out_name = os.path.join(tmp, 'tmalign.out')
        ter_opts = ['-ter', '0'] if all_chains else []
        output = check_output(['TMalign', fname1, fname2, '-m', out_name] + ter_opts)
        rot, trans = parseMatrix(out_name)
    return rot, trans, output


def alignMDTraj(traj1, traj2, legacy=True, all_chains=False):
    """Write trajectories to temporary files, then use runTMalign to align
    them.

    Parameters
    ----------
    traj1 : mdtraj.Trajectory
    traj2 : mdtraj.Trajectory

    Returns
    -------
    rot : numpy.ndarray, shape=(3,3)
        Rotation matrix.
    trans : numpy.ndarray, shape=(3,)
        Translation vector. Should be applied after rot.

    """
    if legacy:
        warnings.warn('Using legacy output format of alignMDTraj')
    traj1 = traj1.atom_slice(traj1.top.select('name CA'))
    traj2 = traj2.atom_slice(traj2.top.select('name CA'))

    with TemporaryDirectory() as tmp:
        fn1 = os.path.join(tmp, 'A.pdb')
        fn2 = os.path.join(tmp, 'B.pdb')
        traj1.save_pdb(fn1)
        traj2.save_pdb(fn2)
        out = runTMalign(fn1, fn2, all_chains=all_chains)
    if legacy:
        return out[:2]
    return out


class Alignment:
    def __init__(self, rot, trans):
        self.rot = rot
        self.trans = trans

    @classmethod
    def from_mdtraj(cls, traj1, traj2):
        rot, trans = alignMDTraj(traj1, traj2, legacy=False)[:2]
        return cls(rot, trans/10)

    def transform_mdtraj(self, traj):
        out = traj.slice(slice(None))
        out.xyz = self.transform(out.xyz)
        return out

    def transform(self, coords):
        return np.inner(coords, self.rot) + self.trans
