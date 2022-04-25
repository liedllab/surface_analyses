#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple
import os.path
import tempfile

import pytest
import numpy as np

from surface_analyses.commandline import main

TrastuzumabRun = namedtuple('TrastuzumabRun', 'scale method expected_data')

ALL_RUNS = [
    (scale, method)
    for scale in ['crippen', 'eisenberg', 'wimley-white']
    for method in ['direct', 'sap', 'sap-byatom', 'potential']
]

BASEPATH = os.path.join(os.path.dirname(__file__), 'trastuzumab')

@pytest.fixture(params=ALL_RUNS)
def trastuzumab_run(request):
    scale, runtype = request.param
    if scale == 'eisenberg':
        scale_fname = 'eisenberg-dg'
    else:
        scale_fname = scale
    with np.load(os.path.join(BASEPATH, scale_fname, f"{runtype}.npz"), allow_pickle=True) as npz:
        yield TrastuzumabRun(scale, runtype, dict(npz))
    return

def run_commandline(parm, traj, scale, outfile, *args):
    main([parm, traj, '--scale', scale, '--out', outfile] + list(args))

def test_output_consistent(trastuzumab_run):
    runtype = trastuzumab_run.method
    scale = trastuzumab_run.scale
    if runtype == 'direct':
        args = ['--surfscore']
    elif runtype == 'sap':
        args = ['--sap', '--surftype', 'sc_norm']
    elif runtype == 'sap-byatom':
        args = ['--sap', '--surftype', 'atom_norm', '--group_heavy']
    elif runtype == 'potential':
        args = ['--potential']
    if scale == 'wimley-white':
        scale = os.path.join(BASEPATH, 'wimley-white-scaled.csv')
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'out.npz')
        parm7 = os.path.join(BASEPATH, 'input.parm7')
        rst7 = os.path.join(BASEPATH, 'input.rst7')
        run_commandline(parm7, rst7, scale, out, *args)
        with np.load(out, allow_pickle=True) as npz:
            assert sorted(npz) == sorted(trastuzumab_run.expected_data)
            for key in npz:
                print(key)
                if key == 'hydrophobic_potential':
                    run_values = npz[key][()]
                    expected = trastuzumab_run.expected_data[key][()]
                    for k, v in run_values.items():
                        np.testing.assert_allclose(v, expected[k])
                elif len(np.asarray(npz[key]).shape) == 2:
                    for i, (v1, v2) in enumerate(zip(npz[key][0], trastuzumab_run.expected_data[key][0])):
                        assert (np.isnan(v1) and np.isnan(v2)) or v1 == pytest.approx(v2), (i, v1, v2)
                else:
                    for i, (v1, v2) in enumerate(zip(npz[key], trastuzumab_run.expected_data[key])):
                        assert v1 == pytest.approx(v2), (i, v1, v2)
                    # np.testing.assert_allclose(npz[key], trastuzumab_run.expected_data[key])
