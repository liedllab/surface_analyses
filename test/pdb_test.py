#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple
import os.path
import subprocess

import pytest
import mdtraj as md
import tempfile

from surface_analyses.pdb import PdbAtom
from surface_analyses.prmtop import RawTopology

TripeptideFiles = namedtuple('TripeptideFiles', 'name pdb parm7 rst7')

def has_tleap():
    try:
        proc = subprocess.run(['tleap', '-h'])
    except FileNotFoundError:
        return False
    if proc.returncode != 0:
        return False
    return True

def get_tripeptide_files(aa, tmp):
    if not has_tleap():
        pytest.skip("tleap not found, or not running successfully")
    leapfile = os.path.join(tmp, 'leap.in')
    pdbfile = os.path.join(tmp, 'tri.pdb')
    parmfile = os.path.join(tmp, 'tri.parm7')
    rstfile = os.path.join(tmp, 'tri.rst7')
    with open(leapfile, 'w') as f:
        print("source leaprc.protein.ff14SB", file=f)
        print("tri = sequence {" + f" N{aa} {aa} C{aa} " +"}", file=f)
        print("check tri", file=f)
        print(f"savepdb tri {pdbfile}", file=f)
        print(f"saveamberparm tri {parmfile} {rstfile}", file=f)
    subprocess.run(['tleap', '-f', leapfile])
    return TripeptideFiles(aa, pdbfile, parmfile, rstfile)


@pytest.fixture(params=['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'CYX', 'GLN', 'GLU',
    'GLY', 'HID', 'HIE', 'HIP', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
    'THR', 'TRP', 'TYR', 'VAL'])
def tripeptide(request):
    with tempfile.TemporaryDirectory() as tmp:
        yield(get_tripeptide_files(request.param, tmp))
    return

@pytest.fixture()
def ala_tripeptide():
    with tempfile.TemporaryDirectory() as tmp:
        yield(get_tripeptide_files("ALA", tmp))
    return

def test_pdb_atom_types_correct(tripeptide):
    pdb_atoms = PdbAtom.list_from_filename(tripeptide.pdb)
    prmtop_atoms = RawTopology.from_file_name(tripeptide.parm7).iter_atoms()
    for pdb_a, prmtop_a in zip(pdb_atoms, prmtop_atoms):
        print(pdb_a, prmtop_a)
        assert pdb_a.atom_type == prmtop_a.atom_type

def test_find_bonded(ala_tripeptide):
    pdb_atoms = PdbAtom.list_from_filename(ala_tripeptide.pdb)
    prmtop_atoms = list(RawTopology.from_file_name(ala_tripeptide.parm7).iter_atoms())
    for atom_list in [pdb_atoms, prmtop_atoms]:
        ca2 = atom_list[14]
        assert ca2.name == "CA"
        assert len(list(ca2.find_bonded())) == 4
        cb = list(ca2.find_bonded(name='CB'))
        assert len(cb) == 1
        assert cb[0].name == 'CB'

def test_residue(ala_tripeptide):
    pdb_atoms = PdbAtom.list_from_filename(ala_tripeptide.pdb)
    prmtop_atoms = list(RawTopology.from_file_name(ala_tripeptide.parm7).iter_atoms())
    for atom_list in [pdb_atoms, prmtop_atoms]:
        cb2 = atom_list[16]
        assert cb2.name == "CB"
        assert len(list(cb2.residue())) == 10

def test_find_in_residue(ala_tripeptide):
    pdb_atoms = PdbAtom.list_from_filename(ala_tripeptide.pdb)
    prmtop_atoms = list(RawTopology.from_file_name(ala_tripeptide.parm7).iter_atoms())
    for atom_list in [pdb_atoms]:
        cb2 = atom_list[16]
        assert cb2.name == "CB"
        assert next(cb2.find_in_residue(name='N')).name == 'N'
