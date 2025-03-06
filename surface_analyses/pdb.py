import abc
import logging
import itertools

import mdtraj as md

from .data import ATOM_DATA
from .amber_compatible_mdtraj_topology import AmberCompatibleTopology


class AbstractAtom(abc.ABC):
    @property
    @abc.abstractmethod
    def i(self):
        return NotImplemented

    @property
    @abc.abstractmethod
    def residue_id(self):
        return NotImplemented

    @property
    @abc.abstractmethod
    def residue_label(self):
        return NotImplemented

    @property
    @abc.abstractmethod
    def name(self):
        return NotImplemented

    @property
    @abc.abstractmethod
    def atomic_number(self):
        return NotImplemented

    @property
    @abc.abstractmethod
    def atom_type(self):
        return NotImplemented

    @property
    def is_heavy(self):
        return self.atomic_number != 1

    def find_bonded(self, **kwargs):
        for other in self.bonded_atoms:
            skip = False
            for k, v in kwargs.items():
                if getattr(other, k) != v:
                    skip = True
                    break
            if not skip:
                yield other

    def residue(self):
        resid = self.residue_id
        residue = set()
        new = {self}
        while new:
            residue.update(new)
            new = set(
                itertools.chain(*[a.find_bonded(residue_id=resid) for a in new])
            ) - residue
        return residue

    def find_in_residue(self, **kwargs):
        residue = self.residue()
        for atom in residue:
            skip = False
            for k, v in kwargs.items():
                if getattr(atom, k) != v:
                    skip = True
                    break
            if not skip:
                yield atom

    def _is_n_terminal_n(self):
        if self.name != "N":
            return False
        bonded = self.bonded_atoms
        for other in bonded:
            if other.name == "C":
                return False
        logging.info(f"atom {self} is an N terminus.")
        return True

    def _is_n_terminal_ca(self):
        if self.name != "CA":
            return False
        n = next(a for a in self.bonded_atoms if a.name == "N")
        if n._is_n_terminal_n():
            return True
        return False

    def _is_c_terminal_c(self):
        if self.name != "C":
            return False
        bonded = self.bonded_atoms
        for other in bonded:
            if other.name == "N":
                return False
        logging.info(f"atom {self} is a C terminus.")
        return True

    def _is_c_terminal_o(self):
        if self.name != "O" and self.name != "OXT":
            return False
        c = self.bonded_atoms[0]
        if c.name != "C":
            raise ValueError(f"Weird atom naming: {c} bonded to {self}")
        return c._is_c_terminal_c()


class PdbAtom(AbstractAtom):

    def __init__(self, i, residue_id, residue_label, name, atomic_number, atom_type=None):
        self._i = i
        self._residue_id = residue_id
        self._residue_label = residue_label
        self._name = name
        self._atomic_number = atomic_number
        self._atom_type = atom_type
        self.bonded_atoms = []

    @property
    def i(self):
        return self._i

    @property
    def residue_id(self):
        return self._residue_id

    @property
    def residue_label(self):
        return self._residue_label

    @property
    def name(self):
        return self._name

    @property
    def atomic_number(self):
        return self._atomic_number

    @property
    def atom_type(self):
        if self._atom_type is not None:
            return self._atom_type
        if self._is_n_terminal_n():
            return "N3"
        if self._is_c_terminal_o():
            return "O2"
        if self.name in ('H1', 'H2', 'H3') and self.residue_label != "ACE":
            return "H"
        if self.name == "HA" and self.bonded_atoms[0]._is_n_terminal_ca():
            return "HP"
        if self.name in ('HA2', 'HA3') and next(self.find_in_residue(name='N'))._is_n_terminal_n():
            return "HP"
        if (
            self.residue_label == "PRO"
            and self.name in ('HD2', 'HD3')
            and next(self.find_in_residue(name='N'))._is_n_terminal_n()
        ):
            return "HP"
        key = (self.residue_label, self.name)
        return ATOM_DATA[key][0]

    def _bond(self, other):
        if not other in self.bonded_atoms:
            self.bonded_atoms.append(other)
        if not self in other.bonded_atoms:
            other.bonded_atoms.append(self)

    @classmethod
    def list_from_filename(cls, filename):
        top = md.load_pdb(filename, standard_names=False).top
        return cls.list_from_md_topology(top)

    @classmethod
    def list_from_md_topology(cls, top: md.Topology):
        top = AmberCompatibleTopology.from_topology(top)
        top.create_standard_bonds()
        out = []
        for at in top.atoms:
            out.append(cls(at.index, at.residue.index, at.residue.name, at.name, at.element.number))
        for at1, at2 in top.bonds:
            out[at1.index]._bond(out[at2.index])
        return out

    def __repr__(self):
        return f'PdbAtom({self.i}, {self.residue_id}, {self.residue_label}, {self.name}, {self.atomic_number}, atom_type={self._atom_type})'
