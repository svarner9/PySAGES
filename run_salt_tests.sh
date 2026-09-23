#!/usr/bin/env bash
# Run this worktree's autoionization-CV tests (task T1, SALT/plan/PAPER_PLAN.md),
# MetaDSpectralABF / Tq / AuxiliaryWall / mlmd-backend-resume tests (task T2,
# SALT/plan/metad_sabf_design.md), and Funnel_MetaDSpectralABF / tq_upper_wall tests (the
# external_force/Frestr follow-up on top of T2) on CPU, with a fixed, persistent interpreter --
# not a session-scoped temp venv.
#
# By default this skips tests marked `slow` (the 1D/2D toy convergence runs in
# tests/test_metad_sabf.py and tests/test_funnel_metad_sabf.py, ~15-40s each) to keep the
# default runtime reasonable; pass `-m slow` as an extra arg to run only those, or `-m ""` to
# run everything.
#
# Interpreter: /scratch/slv5250/tmp/mlmd-clean-env, a persistent CPU-only venv
# (jax/ase/mlmd already installed there for other mlmd work) that this script
# additionally provisions with pysages's lightweight runtime deps (dill,
# plum-dispatch, setuptools_scm -- NOT numba/cython: those are only needed by
# pysages' HOOMD/LAMMPS/ASE-view backends, unused by the mlmd backend or the
# colvars tests here) and an editable install of *this* worktree
# (pysages-mlmd-salt), so `import pysages` resolves here regardless of $PWD
# or $PYTHONPATH. Safe to re-run: every step is idempotent.
#
# Override the interpreter with SALT_TEST_PYTHON if you maintain your own env
# instead (it must be able to `pip install -e` this directory).
#
# Usage: ./run_salt_tests.sh [extra pytest args]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SALT_TEST_PYTHON:-/scratch/slv5250/tmp/mlmd-clean-env/bin/python3}"

if [ ! -x "$PYTHON" ]; then
    echo "run_salt_tests.sh: interpreter not found or not executable: $PYTHON" >&2
    echo "  set SALT_TEST_PYTHON to a python3 that can 'pip install -e' this directory" >&2
    exit 2
fi

# Idempotent provisioning: pysages' runtime deps beyond what mlmd-clean-env
# already carries for mlmd/ase/jax work, plus this worktree itself (editable,
# so edits here are picked up without reinstalling).
"$PYTHON" -m pip install --quiet dill "plum-dispatch>=1.5.4,!=2.0.0,!=2.0.1" setuptools_scm
"$PYTHON" -m pip install --quiet --no-build-isolation --no-deps -e "$HERE"

cd "$HERE"
export JAX_PLATFORMS=cpu
exec "$PYTHON" -m pytest -q -m "not slow" \
    tests/test_autoion_cvs.py tests/test_cvs.py tests/test_nematic_cv.py \
    tests/test_metad_sabf.py tests/test_tq_and_aux_wall.py tests/test_mlmd_backend.py \
    tests/test_funnel_metad_sabf.py \
    "$@"
