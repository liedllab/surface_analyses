"""
More flexible implementation of the Spatial Aggregation Propensity (SAP)
algorithm.

Steps to do a SAP calculation:
    * load a structure. I use mdtraj for the SASA-related functions. The blur
      should be usable with any library, though.
    * assign hydrophobicity values to each atom. assign_propensities can be
      used for that.
    * compute the solvent-accessible surface (SAA), e.g., using
      mdtraj.shrake_rupley
    * Apply a blur. The default SAP algorithm uses a hard distance cutoff of 5
      Å. A weighting function can be used for blur, but this is not supported
      in the command line interface.

This implementation can be used from a script/Jupyter notebook, or via the
command line interface. The functionality of the CLI is limited, however. It
supports a single PDB input. The distance cutoff can be changed, but no
weighting functions are implemented.
"""
import mdtraj as md
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Callable, Optional
from scipy.spatial import cKDTree


def blur(
    xyz: Union[np.ndarray, List[List[float]]],
    v: Union[np.ndarray, List],
    rmax: float,
    weight_fun: Optional[Callable] = None
):
    xyz = np.asarray(xyz)
    if len(xyz.shape) != 2:
        raise ValueError("Expected 2D coordinates, but shape is " + str(xyz.shape))
    v = np.asarray(v)

    tree = cKDTree(xyz)
    blurred: List[float] = []
    neighbors_per_atom = tree.query_ball_tree(tree, rmax)
    for point, neighbors in zip(xyz, neighbors_per_atom):
        if weight_fun is not None:
            dist = np.sqrt(((xyz[neighbors, :] - point) ** 2).sum(1))
            weights = weight_fun(dist)
        else:
            weights = 1
        blurred.append((v[neighbors] * weights).sum())
    return np.array(blurred)


# Sidechain SAA in nm^2, averaged over short MD simulations of ACE-X-NME.
_default_sidechain_saa = {
    "ALA": 0.594017550349,
    "ARG": 2.05252944492,
    "ASN": 1.09079792118,
    "ASP": 0.971595257521,
    "CYS": 0.882727324963,
    "GLN": 1.38791826623,
    "GLU": 1.30966766085,
    "GLY": 0.186710074544,
    "HIS": 1.416356848696,  # 88.8 % HID/HIE, 11.2 % HIP
    "HID": 1.41232917458,  # twice the same for HID and HIE
    "HIE": 1.41232917458,
    "HIP": 1.44829055062,
    "ILE": 1.33404062502,
    "LEU": 1.40599354915,
    "LYS": 1.71556117944,
    "MET": 1.48098336253,
    "PHE": 1.68156359531,
    "PRO": 1.12518100068,
    "SER": 0.742956539616,
    "THR": 1.05462151393,
    "TRP": 2.03259896627,
    "TYR": 1.82585927006,
    "VAL": 1.10587068833,
    "NME": 100000., # high value => always counts as buried.
    "ACE": 100000.,
}
