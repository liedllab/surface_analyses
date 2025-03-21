import pytest
import numpy as np

import surface_analyses.surface as surf
import surface_analyses.patches as patches

@pytest.fixture
def minimal():
    vertices = [[0., 0., 0.], [0., 0., 1.], [0., 1., 1.], [0., 2., 1.], [0., 2., 2.]]
    faces = [[0, 1, 2], [2, 3, 4]]
    s = surf.Surface(vertices, faces)
    s.data['DATA'] = [-2, -1, 0, 1, 2]
    return s

def test_finds_patches_full(minimal):
    p = patches.find_patches(minimal.faces, [True, True, True, True, True])
    assert len(p) == 1
    assert np.all(p[0] == [0, 1, 2, 3, 4])

def test_finds_patches_separate(minimal):
    p = patches.find_patches(minimal.faces, [True, True, False, True, True])
    assert len(p) == 2
    assert np.all(p[0] == [0, 1])
    assert np.all(p[1] == [3, 4])
    area = minimal.vertex_areas()
    assert area[p[0]].sum() == 1.0

def test_finds_patches_separate(minimal):
    p = patches.find_patches(minimal.faces, [True, True, False, True, True])
    assert len(p) == 2
    assert np.all(p[0] == [0, 1])
    assert np.all(p[1] == [3, 4])
    area = minimal.vertex_areas()
    assert area[p[0]].sum() == pytest.approx(1/3)
    assert area[p[1]].sum() == pytest.approx(1/3)
