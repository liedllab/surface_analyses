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
