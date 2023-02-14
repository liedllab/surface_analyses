#!/usr/bin/env python
# -*- coding: utf-8 -*-

from surface_analyses.ele_patches import main
from contextlib import redirect_stdout
import io
from pathlib import Path
import os

import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np
import pytest


TESTS_PATH = Path(os.path.dirname(__file__))
TRASTUZUMAB_PATH = TESTS_PATH / 'trastuzumab'


def run_commandline(pdb, dx, *args, surface_type="sas"):
    output = io.StringIO()
    with redirect_stdout(output):
        main([str(pdb), str(dx), "--surface_type", surface_type, '--check_cdrs'] + list(args))
    return output.getvalue()


def test_trastuzumab_sas_integrals():
    expected = np.array(
        [
            22575.20631872,
            12389.29284473,
            27131.66874305,
            -4556.46242433,
            -1867.35195722,
        ]
    )
    out_lines = run_commandline(
        TRASTUZUMAB_PATH / "apbs-input.pdb",
        TRASTUZUMAB_PATH / "apbs-potential.dx",
        '--out',
        str(TRASTUZUMAB_PATH / 'apbs-patches.csv'),
        surface_type="sas",
    )
    last = out_lines.splitlines()[-1]
    integrals = np.array([float(x) for x in last.split()])
    assert np.allclose(expected, integrals)
    patches = pd.read_csv(str(TRASTUZUMAB_PATH / 'apbs-patches.csv'))
    expected_patches = pd.read_csv(str(TRASTUZUMAB_PATH / 'apbs-patches.save'))
    assert_frame_equal(patches, expected_patches)
