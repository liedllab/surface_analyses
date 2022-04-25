from collections import ChainMap

import pytest

import surface_analyses.crippen as crippen
import surface_analyses.pdb as pdb

REGULAR_ATOMS = {
    "gly0_c": pdb.PdbAtom(0, 1, "GLY", "C", 6, "C"),
    "gln1_n": pdb.PdbAtom(1, 2, "GLN", "N", 7, "N"),
}

TERMINUS_ATOMS = {
    "ace0_c": pdb.PdbAtom(0, 1, "ACE", "C", 8, "C"),
    "arg0_cterm_o": pdb.PdbAtom(0, 1, "ARG", "OXT", 8, "O"),
    "arg0_nterm_n": pdb.PdbAtom(0, 1, "ARG", "N", 7, "N"),
    "arg0_nterm_h1": pdb.PdbAtom(0, 1, "ARG", "H1", 1, "H"),
    "arg0_nterm_h2": pdb.PdbAtom(0, 1, "ARG", "H2", 1, "H"),
    "arg0_nterm_h3": pdb.PdbAtom(0, 1, "ARG", "H3", 1, "H"),
    "arg0_nterm_ca": pdb.PdbAtom(0, 1, "ARG", "CA", 6, "C"),
}

TERMINUS_ATOMS['arg0_nterm_n']._bond(TERMINUS_ATOMS["arg0_nterm_h1"])
TERMINUS_ATOMS['arg0_nterm_n']._bond(TERMINUS_ATOMS["arg0_nterm_h2"])
TERMINUS_ATOMS['arg0_nterm_n']._bond(TERMINUS_ATOMS["arg0_nterm_h3"])
TERMINUS_ATOMS['arg0_nterm_n']._bond(TERMINUS_ATOMS["arg0_nterm_ca"])

CORRECT_TYPE = {
    "gly0_c": "C5",
    "gln1_n": "N2",
    "ace0_c": "C1",
    "arg0_cterm_o": "O12",
    'arg0_nterm_n': "N10",
    'arg0_nterm_h1': "H3",
    'arg0_nterm_h2': "H3",
    'arg0_nterm_h3': "H3",
    'arg0_nterm_ca': "C4",
}

ALL_EXAMPLE_ATOMS = ChainMap(REGULAR_ATOMS, TERMINUS_ATOMS)

@pytest.fixture(params=list(ALL_EXAMPLE_ATOMS))
def any_atom(request):
    return request.param

def test_crippen_type(any_atom):
    atom = ALL_EXAMPLE_ATOMS[any_atom]
    assert crippen.crippen_type(atom, fix_termini=True) == CORRECT_TYPE[any_atom]

def test_logp_does_not_raise(any_atom):
    crippen.logp(ALL_EXAMPLE_ATOMS[any_atom])
