#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple
from pathlib import Path
import os.path
import shutil
import tempfile

import pytest
import numpy as np

from surface_analyses.commandline import main

TrastuzumabRun = namedtuple('TrastuzumabRun', 'scale method expected_data')

ALL_RUNS = [
    (scale, method)
    for scale in ['crippen', 'eisenberg', 'wimley-white']
    for method in ['direct', 'sap', 'sap-byatom', 'potential', 'surrounding-hydrophobicity']
]

TESTS_PATH = Path(os.path.dirname(__file__))
TRASTUZUMAB_PATH = TESTS_PATH / 'trastuzumab'

CsA_SMILES = 'CCC1C(=O)N(CC(=O)N(C(C(=O)NC(C(=O)N(C(C(=O)NC(C(=O)NC(C(=O)N(C(C(=O)N(C(C(=O)N(C(C(=O)N(C(C(=O)N1)C(C(C)CC=CC)O)C)C(C)C)C)CC(C)C)C)CC(C)C)C)C)C)CC(C)C)C)C(C)C)CC(C)C)C)C'

@pytest.fixture(params=ALL_RUNS)
def trastuzumab_run(request):
    scale, runtype = request.param
    if scale == 'eisenberg':
        scale_fname = 'eisenberg-dg'
    else:
        scale_fname = scale
    with np.load(TRASTUZUMAB_PATH / scale_fname / f"{runtype}.npz", allow_pickle=True) as npz:
        yield TrastuzumabRun(scale, runtype, dict(npz))
    return

def run_commandline(parm, traj, scale, outfile, *args):
    main([str(parm), str(traj), '--scale', str(scale), '--out', str(outfile)] + list(args))

def test_output_consistent(trastuzumab_run):
    runtype = trastuzumab_run.method
    if runtype == 'potential' and shutil.which('TMalign') is None:
        pytest.skip('TMalign was not found')
    scale = trastuzumab_run.scale
    parm7 = TRASTUZUMAB_PATH / 'input.parm7'
    rst7 = TRASTUZUMAB_PATH / 'input.rst7'
    ref = TRASTUZUMAB_PATH / '1f4w-standard-orientation.pdb'
    print(f'{runtype=}')
    if runtype == 'direct':
        args = ['--surfscore']
    elif runtype == 'sap':
        args = ['--sap', '--surftype', 'sc_norm']
    elif runtype == 'sap-byatom':
        args = ['--sap', '--surftype', 'atom_norm', '--group_heavy']
    elif runtype == 'potential':
        args = ['--potential', '--ref', str(ref)]
    elif runtype == 'surrounding-hydrophobicity':
        args = ['--sh']
    else:
        raise ValueError(runtype)

    if scale == 'wimley-white':
        scale = TRASTUZUMAB_PATH / 'wimley-white-scaled.csv'
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / 'out.npz'
        run_commandline(parm7, rst7, scale, out, *args)
        with np.load(out, allow_pickle=True) as npz:
            assert_outputs_equal(npz, trastuzumab_run.expected_data)


def test_output_with_sc_norm():
    scale = TRASTUZUMAB_PATH / 'glmnet.csv'
    args = ['--surfscore', '--surftype', 'sc_norm']
    parm7 = TRASTUZUMAB_PATH / 'input.parm7'
    rst7 = TRASTUZUMAB_PATH / 'input.rst7'
    expected_fname = TRASTUZUMAB_PATH / 'jain-surfscore-sc-norm.npz'
    with np.load(expected_fname, allow_pickle=True) as npz:
        expected = dict(npz)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.npz"
        run_commandline(parm7, rst7, scale, out, *args)
        with np.load(out, allow_pickle=True) as npz:
            assert_outputs_equal(npz, expected)


def test_rdkit_crippen():
    pdb = TESTS_PATH / '1csa-model1.pdb'
    args = ['--surfscore', '--smiles', CsA_SMILES]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.npz"
        run_commandline(pdb, pdb, 'rdkit-crippen', out, *args)
        with np.load(out, allow_pickle=True) as npz:
            assert np.sum(npz['propensities']) == pytest.approx(3.269)


def assert_outputs_equal(a, b):
    """Given two dicts or npz files containing a surfscore output file, assert
    that they are equal."""
    assert sorted(a) == sorted(b)
    for key in a:
        print(key)
        if key == 'hydrophobic_potential':
            run_values = a[key][()]
            expected = b[key][()]
            for k, v in run_values.items():
                print(f"hydrophobic pot: {k}")
                v_exp = expected[k]
                assert np.mean(v) == pytest.approx(np.mean(v_exp), rel=1e-2, abs=1e-4)
                assert np.std(v) == pytest.approx(np.std(v_exp), rel=1e-2, abs=1e-4)
        elif len(np.asarray(a[key]).shape) == 2:
            for frame in range(len(a[key])):
                np.testing.assert_allclose(a[key][frame], b[key][frame])
        else:
            for i, (v1, v2) in enumerate(zip(a[key], b[key])):
                assert v1 == pytest.approx(v2), (i, v1, v2)
