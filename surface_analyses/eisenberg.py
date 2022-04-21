#!/bin/env python

import prmtop.raw_topology as raw_top

EISENBERG_TYPES = {
    "C":  "C",
    "CA": "C",
    "CB": "C",
    "CC": "C",
    "CK": "C",
    "CM": "C",
    "CN": "C",
    "CQ": "C",
    "CR": "C",
    "CT": "C",
    "CV": "C",
    "CW": "C",
    "CX": "C",
    "C*": "C",
    "2C": "C",
    "3C": "C",
    "C8": "C",
    "CO": "C",
    "N":  "N/O",
    "NA": "N/O",
    "NB": "N/O",
    "NC": "N/C",
    "NT": None, #"sp2 nitrogen with 3 substituents",
    "N2": "N+", # ARG
    "N3": "N+", # LYS, LYN (!!!), C-term
    "N*": None, #"sp2 nitrogen in purine or pyrimidine with alkyl group attached",
    "O": "N/O",
    "OH": "N/O",
    "OS": None, # "ether or ester oxygen",
    "OW": None, # "water oxygen",
    "O2": "O-",
    "S": "S",
    "SH": "S",
}

EISENBERG_PARAMS = {  # unit: cal A^-2 mol^-1
    'C': 16, # ±2
    'N/O': -6, # ±4
    'O-': -24, # ±10
    'N+': -50, # ±9
    'S': 21, # ±10
}
EISENBERG_PARAMS['N+1/2'] = (EISENBERG_PARAMS['N/O'] + EISENBERG_PARAMS['N+']) / 2

def heavy_atoms(top: raw_top.RawTopology, nmax=None):
    for at in top.iter_atoms(nmax):
        if at.atomic_number != 1:
            yield at

def amber_to_eisen_type(atom):
    return eisen_type(atom.atom_type, atom.name, atom.residue_label)

def amber_to_eisen_value(atom):
    return EISENBERG_PARAMS[amber_to_eisen_type(atom)]

def eisen_type(type, name, residue):
    typ = EISENBERG_TYPES[type]
    if residue == 'ARG':
        if name == 'NE':
            typ = 'N/O'
        elif name in ('NH1', 'NH2'):
            typ = 'N+1/2'
    return typ