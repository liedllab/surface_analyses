import warnings

from .data import ATOM_DATA, CRIPPEN_PARAMS


def crippen_type(atom, fix_termini=True, typemap=None):
    """Assign crippen atom type to an atom.
    
    If fix_termini is True, attempt to assign types to non-capped, charged termini. 
    Assumes that all atoms are present.
    
    Data taken from RdKit:
    https://github.com/rdkit/rdkit/blob/7a5491276511f239b743bbfef0bd06b4f359d07a/Data/Crippen.txt
    """
    if fix_termini:
        try:
            return _fix_terminus_type(atom)
        except ValueError:
            pass
    if typemap is None:
        return _raw_crippen_type(atom.residue_label, atom.name)
    else:
        return typemap[atom.residue_label, atom.name]


def _raw_crippen_type(residue, atom):
    return ATOM_DATA[(residue, atom)][1]


def _fix_terminus_type(atom):
    # C-terminus
    if atom.name == 'OXT':
        return 'O12'
    # N-terminal N
    if atom.name == 'N':
        bonded = atom.bonded_atoms
        if len(bonded) == 4:
            return 'N10'
    # hydrogen bonded to nitrogen (e.g., N-terminal H)
    if atom.atomic_number == 1:
        bonded = atom.bonded_atoms
        if not bonded:
            warnings.warn('Treating unbonded hydrogen as H3')
            return 'H3'
        if bonded and bonded[0].name == 'N':
            return 'H3'
    raise ValueError("Atom does not match a known special case for Crippen assignment.")


def logp(atom, fix_termini=True, typemap=None):
    type = crippen_type(atom, fix_termini=fix_termini, typemap=typemap)
    return CRIPPEN_PARAMS[type][0]
