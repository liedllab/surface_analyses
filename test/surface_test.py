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
    plydat = minimal.as_plydata(nanometers_per_length_unit=1.)
    assert len(plydat['vertex']) == minimal.n_vertices
    for i_dim, dim in enumerate('xyz'):
        assert np.allclose(plydat['vertex'][dim], minimal.vertices[:, i_dim])

def test_write_ply(minimal):
    minimal.write_ply('test.ply')
