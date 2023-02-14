import pytest
import numpy as np

import surface_analyses.hydrophobic_potential as hp
import gisttools as gt

def test_grid_with_walldist():
    grid = hp.grid_with_walldist([[-4, 0, 1], [4, 0, 1.1]], 2, 0.5)
    assert np.allclose(grid.origin, [-5.75, -1.75, -0.75])
    assert np.all(grid.xyzmax + 0.5 >= [6, 2, 3.1])
