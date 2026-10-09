#!/usr/bin/env bash
# Create .venv and install dependencies — works fully OFFLINE.
#
#   offline/cpython-3.12-x86_64-linux-gnu.tar.gz : portable CPython 3.12 (python-build-standalone)
#   offline/wheels/                               : all packages as manylinux x86_64 cp312 wheels
#
# Default: use the bundled Python (no dependence on the server's Python version).
#   USE_SYSTEM_PYTHON=1 PYTHON=python3.12 ./setup_env.sh   -> use a system Python 3.12 instead
# Without the offline/ folder it falls back to installing from the internet.
set -euo pipefail
cd "$(dirname "$0")"

[[ $(uname -m) == x86_64 ]] || { echo "ERROR: bundle is for x86_64, this machine is $(uname -m)"; exit 1; }

if [[ -z "${USE_SYSTEM_PYTHON:-}" && -f offline/cpython-3.12-x86_64-linux-gnu.tar.gz ]]; then
  if [[ ! -x offline/python/bin/python3 ]]; then
    echo "extracting bundled CPython 3.12 -> offline/python"
    tar xzf offline/cpython-3.12-x86_64-linux-gnu.tar.gz -C offline     # creates offline/python/
  fi
  PYTHON=$PWD/offline/python/bin/python3
else
  PYTHON=${PYTHON:-python3}
fi
echo "Base interpreter: $PYTHON ($($PYTHON --version))"

if [[ ! -x .venv/bin/python ]]; then
  $PYTHON -m venv .venv 2>/dev/null || $PYTHON -m venv --without-pip .venv
fi

if [[ -d offline/wheels ]]; then
  echo "installing from offline/wheels (no network)"
  if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    # bootstrap pip from its own wheel
    .venv/bin/python offline/wheels/pip-*.whl/pip install --no-index offline/wheels/pip-*.whl
  fi
  .venv/bin/python -m pip install -q --no-index --find-links offline/wheels -r requirements.txt
else
  echo "offline/wheels not found -> installing from the internet"
  if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    curl -sSL https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
  fi
  .venv/bin/python -m pip install -q -r requirements.txt
fi

.venv/bin/python -c "import faiss, numba, numpy, sklearn, pyarrow, pandas, matplotlib, psutil; print('OK  faiss', faiss.__version__, '| numba', numba.__version__, '| numpy', numpy.__version__)"
command -v fincore >/dev/null && echo "fincore: available" \
  || echo "fincore: NOT installed (extra credit falls back to mincore(2), same syscall; install util-linux-extra if you can)"
