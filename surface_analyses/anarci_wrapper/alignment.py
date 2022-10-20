from .annotation import Annotation

_CORESET = {
    'light': [
        44, 19, 69, 14, 75, 82, 15, 21, 47, 20, 48, 49, 22, 81, 79, 80, 23, 36,
        35, 37, 74, 88, 38, 18, 87, 17, 86, 85, 46, 70, 45, 16, 71, 72, 73,
    ],
    'heavy': [
        35, 12, 38, 36, 83, 19, 94, 37, 11, 47, 39, 93, 46, 45, 68, 69, 71, 70,
        17, 72, 92, 84, 91, 90, 20, 21, 85, 25, 24, 86, 89, 88, 87, 22, 23,
    ]
}


def select_coreset(traj, coreset=_CORESET):
    """Select CA atoms from an mdtraj.Trajectory and the ABangle coreset."""
    ann = Annotation.from_traj(traj)
    atoms = []
    for chain, residues in coreset.items():
        for res in residues:
            (index,) = ann.chain(chain)[res]
            sel = traj.top.select(f'name CA and resid {index}')
            if len(sel) == 0:
                raise RuntimeError('Missing coreset CA atom: ' + str(index))
            elif len(sel) != 1:
                raise RuntimeError('Non-unique selection occured in select_coreset')
            atoms.append(sel[0])
    return atoms


def get_fv_alignment(a, b, coreset=_CORESET):
    """Get a transformation object that aligns a to b, assuming both are Fv regions.

    Parameters
    ----------
    a, b : mdtraj.Trajectory objects
        Single frame trajectory objects containing the 2 structures to align. Only the
        first frame will be used.
    coreset
        coreset as returned by read_coreset.

    Returns
    -------
    An mdtraj.orientation.Transformation object
    """
    from mdtraj.geometry.alignment import compute_transformation

    a_cut = a.atom_slice(select_coreset(a, coreset))
    b_cut = b.atom_slice(select_coreset(b, coreset))

    return compute_transformation(a_cut.xyz[0], b_cut.xyz[0])


def align_antibodies(a, b, coreset=_CORESET):
    """Return a copy of Fv region a that is aligned to b.

    Uses only the first frame.
    """
    # Copy of the 1st frame => we don't modify the original *a*.
    out = a[0]
    transform = get_fv_alignment(out, b, coreset)
    out.xyz = transform.transform(out.xyz[0]).reshape(1, -1, 3)
    return out
