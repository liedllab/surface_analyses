from gisttools.grid import Grid
import pytest
import numpy as np

import surface_analyses.surface as surf

@pytest.fixture
def minimal():
    s = surf.Surface([[0., 0., 0.], [0., 0., 1.], [0., 1., 1.]], [[0, 1, 2]])
    s.data['DATA'] = [-5, 0, 5]
    return s

def test_create_surf(minimal):
    pass

def test_repr_no_error(minimal):
    repr(minimal)

def test_convert_to_ply(minimal):
    plydat = minimal.as_plydata(units_per_angstrom=1.)
    assert len(plydat['vertex']) == minimal.n_vertices
    for i_dim, dim in enumerate('xyz'):
        assert np.allclose(plydat['vertex'][dim], minimal.vertices[:, i_dim])

def test_write_ply(minimal):
    minimal.write_ply('test.ply')

def test_quantile_skipping_centered_norm():
    norm = surf.QuantileSkippingCenteredNorm(clip_fraction=0.1)
    norm.autoscale(np.arange(-4, 12))
    assert norm.vmax == 10
    assert norm.vmin == -10

def test_gaussian_grid_variable_sigma():
    grid = Grid([-1, 0, 0], [3, 1, 1], 1)
    out = surf.gaussian_grid_variable_sigma(grid, [[-0.5, 0, 0]], [0.3], [2])
    assert np.allclose(out.ravel(), np.exp([-0.5**2/2/0.3**2, -0.5**2/2/0.3**2, -1.5**2/2/0.3**2]))
