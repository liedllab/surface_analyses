#!/usr/bin/env python
# -*- coding: utf-8 -*-

from surface_analyses.commandline_electrostatic import main, biggest_residue_contribution, parse_args, run_electrostatics
from contextlib import redirect_stdout
import inspect
import io
from pathlib import Path
import os
import shutil

import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np
import pytest
import msms.wrapper as msms

try:
    import anarci
    HAS_ANARCI = True
except ImportError:
    HAS_ANARCI = False


TESTS_PATH = Path(os.path.dirname(__file__))
TRASTUZUMAB_PATH = TESTS_PATH / 'trastuzumab'


def get_parameter_list(fun):
    "Return all parameters that can be passed to a function."
    params = inspect.signature(fun).parameters
    return dict(params)


def run_commandline(pdb, dx, *args, **kwargs):
    output = io.StringIO()
    kwargs_list = []
    for k, v in kwargs.items():
        kwargs_list.extend(["--" + str(k), str(v)])
    print(kwargs_list)
    with redirect_stdout(output):
        # using the pdb as "topology" and "trajectory"
        main([str(pdb), str(pdb), "--dx", str(dx)] + list(args) + kwargs_list)
    return output.getvalue()


def test_parser_options_match_python_interface():
    parser_options = vars(parse_args("top.parm7 traj.nc".split()))
    parser_options.pop('parm')
    parser_options.pop('trajs')
    parser_options.pop('ref')
    parser_options.pop('protein_ref')
    parser_options.pop('stride')
    python_options = get_parameter_list(run_electrostatics)
    python_options.pop('traj')
    assert set(python_options) == set(parser_options)
    for opt in parser_options:
        assert parser_options[opt] == python_options[opt].default

@pytest.fixture(params=['without', 'with'])
def with_or_without_cdrs(request):
    yield request.param


def test_trastuzumab_sas_integrals(with_or_without_cdrs):
    expected = np.array(
        [
            25015.40424103,
            12573.01033872,
            29718.71768997,
            -4703.31344894,
            -1867.65861091,
        ]
    )
    if with_or_without_cdrs == 'with':
        if shutil.which('hmmscan') is None:
            pytest.skip('hmmscan was not found')
        if not HAS_ANARCI:
            pytest.skip('ANARCI is not available')
        args = ['--check_cdrs']
    else:
        args = []
    stdout = run_commandline(
        TRASTUZUMAB_PATH / 'apbs-input.pdb',
        TRASTUZUMAB_PATH / 'apbs-potential.dx',
        *args,
        out=TRASTUZUMAB_PATH / 'apbs-patches.csv',
        surface_type='sas',
    )
    print(stdout)
    last = stdout.splitlines()[-1]
    integrals = np.array([float(x) for x in last.split()])
    assert np.allclose(expected, integrals)
    patches = pd.read_csv(str(TRASTUZUMAB_PATH / 'apbs-patches.csv'))
    exp_fname = 'apbs-patches-msms.save' if msms.msms_available() else 'apbs-patches.save'
    expected_patches = pd.read_csv(str(TRASTUZUMAB_PATH / exp_fname))
    if with_or_without_cdrs == 'without':
        expected_patches['cdr'] = False
    print(expected_patches['cdr'].sum(), patches['cdr'].sum())
    assert_frame_equal(patches, expected_patches)


def get_nth_line(file, n):
    file = iter(file)
    for _ in range(n):
        next(file)
    return next(file)


def test_trastuzumab_ply_out():
    args = [
        TRASTUZUMAB_PATH / 'apbs-input.pdb',
        TRASTUZUMAB_PATH / 'apbs-potential.dx',
        '--out',
        str(TRASTUZUMAB_PATH / 'apbs-patches.csv'),
        '--ply_out',
        str(TRASTUZUMAB_PATH / 'apbs'),
    ]
    run_commandline(*args, surface_type="sas")
    # check the number of vertices in the output
    with open(TRASTUZUMAB_PATH / 'apbs-potential.ply') as f:
        if msms.msms_available():
            assert get_nth_line(f, 2).strip() == "element vertex 43411"
        else:
            assert get_nth_line(f, 2).strip() == "element vertex 67252"


def test_biggest_residue_contribution():
    df = pd.DataFrame({
        "residue": ["c", "a", "b", "a"],
        "area": [1, 2, 3, 2],
    })
    assert biggest_residue_contribution(df) == "a"
