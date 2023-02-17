#!/usr/bin/env python
# -*- coding: utf-8 -*-

from surface_analyses.ele_patches import main
from contextlib import redirect_stdout
import io
from pathlib import Path
import os
import shutil

import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np
import pytest


TESTS_PATH = Path(os.path.dirname(__file__))
TRASTUZUMAB_PATH = TESTS_PATH / 'trastuzumab'


def run_commandline(pdb, dx, *args, surface_type="sas"):
    output = io.StringIO()
    with redirect_stdout(output):
        main([str(pdb), str(dx), "--surface_type", surface_type] + list(args))
    return output.getvalue()


@pytest.fixture(params=['with', 'without'])
def with_or_without_cdrs(request):
    yield request.param


def test_trastuzumab_sas_integrals(with_or_without_cdrs):
    expected = np.array(
        [
            22575.20631872,
            12389.29284473,
            27131.66874305,
            -4556.46242433,
            -1867.35195722,
        ]
    )
    args = [
        TRASTUZUMAB_PATH / 'apbs-input.pdb',
        TRASTUZUMAB_PATH / 'apbs-potential.dx',
        '--out',
        str(TRASTUZUMAB_PATH / 'apbs-patches.csv'),
    ]
    kwargs = {'surface_type': 'sas'}
    if with_or_without_cdrs == 'with':
        if shutil.which('hmmscan') is None:
            pytest.skip('hmmscan was not found')
        args.append('--check_crds')
    out_lines = run_commandline(*args, **kwargs)
    last = out_lines.splitlines()[-1]
    integrals = np.array([float(x) for x in last.split()])
    assert np.allclose(expected, integrals)
    patches = pd.read_csv(str(TRASTUZUMAB_PATH / 'apbs-patches.csv'))
    expected_patches = pd.read_csv(str(TRASTUZUMAB_PATH / 'apbs-patches.save'))
    if with_or_without_cdrs == 'without':
        expected_patches['cdr'] = False
    assert_frame_equal(patches, expected_patches)
