#!/usr/bin/env python
# -*- coding: utf-8 -*-

from surface_analyses.commandline_electrostatic import main, biggest_residue_contribution, parse_args, run_electrostatics
from surface_analyses.surface import Surface

from contextlib import redirect_stdout
import inspect
import io
from pathlib import Path
import os
import shutil
from unittest.mock import patch
from tempfile import TemporaryDirectory

import mdtraj as md
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
    args_list = [str(arg) for arg in args]
    with redirect_stdout(output):
        # using the pdb as "topology" and "trajectory"
        main([str(pdb), str(pdb), "--dx", str(dx)] + args_list + kwargs_list)
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


@pytest.fixture(params=[[], ['--check_cdrs']])
def with_or_without_cdrs(request):
    yield request.param


def test_trastuzumab_sas_integrals(with_or_without_cdrs):
    expected = np.array(
        [
            25.01540424103,
            12.57301033872,
            29.71871768997,
            -4.70331344894,
            -1.86765861091,
        ]
    )
    resout_fname = TRASTUZUMAB_PATH / 'resout.csv'
    args = ['--resout', resout_fname] + with_or_without_cdrs
    if "--check_cdrs" in args:
        if shutil.which('hmmscan') is None:
            pytest.skip('hmmscan was not found')
        if not HAS_ANARCI:
            pytest.skip('ANARCI is not available')
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
    if "--check_cdrs" not in args:
        del expected_patches['cdr']
    assert_frame_equal(patches, expected_patches)
    resout_df = pd.read_csv(resout_fname)
    expected_n_patches = 32 if msms.msms_available() else 36
    assert len(resout_df) == expected_n_patches
    assert resout_df.iloc[0]["residues"].startswith(
        "TYR33 ARG50 ASN55 TYR57 THR58 ARG59 TYR60 ALA61 ASP62 LYS65 GLY66 TRP99 "
        "ASP122 ILE123 GLN148 HIS212 TYR213 THR214 THR215 PRO216"  # PRO217 is missing w/o msms
    )


def get_nth_line(file, n):
    file = iter(file)
    for _ in range(n):
        next(file)
    return next(file)


def test_trastuzumab_ply_out():
    with TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        pdb_name = str(TRASTUZUMAB_PATH / 'apbs-input.pdb')
        pot_name = str(TRASTUZUMAB_PATH / 'apbs-potential.dx')
        ply_name = str(tmp / 'apbs')
        csv_name = str(tmp / 'apbs-patches-ply.csv')
        args = [pdb_name, pot_name, '--out', csv_name, '--ply_out', ply_name]
        run_commandline(*args, surface_type='sas')
        # check the number of vertices in the output
        with open(ply_name + '-potential.ply') as f:
            if msms.msms_available():
                assert get_nth_line(f, 2).strip() == 'element vertex 40549'
            else:
                assert get_nth_line(f, 2).strip() == 'element vertex 67252'
        surf = Surface.read_ply(ply_name + '-potential.ply')
        crd = md.load(pdb_name).xyz[0]
        surf_extent = surf.vertices.max(axis=0) - surf.vertices.min(axis=0)
        crd_extent = crd.max(axis=0) - crd.min(axis=0)
        scale_ratios = surf_extent / crd_extent
        assert np.all(scale_ratios > 1)
        assert np.all(scale_ratios < 1.2)



def test_biggest_residue_contribution():
    df = pd.DataFrame({
        "residue": ["c", "a", "b", "a"],
        "area": [1, 2, 3, 2],
    })
    assert biggest_residue_contribution(df) == "a"
