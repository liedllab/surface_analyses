import abc
from functools import cache
import csv

import mdtraj as md

from .data import ATOM_DATA

def amber_type(res, atom):
    try:
        return ATOM_DATA[(res, atom)][0]
    except KeyError:
        try:
            return ATOM_DATA[('C' + res, atom)][0]
        except KeyError:
            return ATOM_DATA[('N' + res, atom)][0]


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


@AbstractAtom.register
class PdbAtom:

    def __init__(self, i, residue_id, residue_label, name, atomic_number, atom_type=None):
        self.i = i
        self.residue_id = residue_id
        self.residue_label = residue_label
        self.name = name
        self.atomic_number = atomic_number
        if atom_type is None:
            atom_type = amber_type(residue_label, name)
        self.atom_type = atom_type
        self.bonded_atoms = []

    def _bond(self, other):
        if not other in self.bonded_atoms:
            self.bonded_atoms.append(other)
        if not self in other.bonded_atoms:
            other.bonded_atoms.append(self)

    @classmethod
    def list_from_md_topology(cls, top: md.Topology):
        out = []
        for at in top.atoms:
            out.append(cls(at.index, at.residue.index, at.residue.name, at.name, at.element.number))
        for at1, at2 in top.bonds:
            out[at1.index]._bond(out[at2.index])
        return out

    def __repr__(self):
        return f'PdbAtom({self.i=}, {self.residue_id=}, {self.residue_label=}, {self.name=}, {self.atomic_number=}, {self.atom_type=})'
