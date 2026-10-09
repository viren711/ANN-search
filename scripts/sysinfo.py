"""Record the machine the experiments ran on (needed for the report): CPU, RAM, disks, GPUs,
OS, Python and package versions. Output: results/system/."""
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import psutil

from ann.bench import cpu_model
from ann.data import RESULTS


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout
    except Exception as e:                               # noqa: BLE001
        return f"<failed: {e}>"


out = RESULTS / "system"
out.mkdir(parents=True, exist_ok=True)
dumps = {"lscpu.txt": "lscpu", "free.txt": "free -h", "lsblk.txt": "lsblk -o NAME,MODEL,ROTA,TRAN,SIZE,MOUNTPOINT",
         "df.txt": "df -h", "nvidia-smi.txt": "nvidia-smi", "pip-freeze.txt": f"{sys.executable} -m pip freeze",
         "uname.txt": "uname -a", "numa.txt": "numactl --hardware"}
for f, c in dumps.items():
    (out / f).write_text(sh(c))

import faiss  # noqa: E402
import numba  # noqa: E402
import numpy  # noqa: E402
import sklearn  # noqa: E402

info = {"hostname": platform.node(), "cpu_model": cpu_model(), "logical_cpus": os.cpu_count(),
        "physical_cores": psutil.cpu_count(logical=False),
        "ram_gib": round(psutil.virtual_memory().total / 2**30, 1), "python": sys.version.split()[0],
        "os": platform.platform(), "faiss": faiss.__version__, "numba": numba.__version__,
        "numba_threads": numba.config.NUMBA_NUM_THREADS, "numpy": numpy.__version__,
        "sklearn": sklearn.__version__, "fincore_cli": bool(sh("command -v fincore").strip()),
        "env": {k: os.environ.get(k) for k in ["BUILD_THREADS", "SEARCH_THREADS", "NUMBA_NUM_THREADS",
                                               "OMP_NUM_THREADS"]}}
(out / "system.json").write_text(json.dumps(info, indent=2))
print(json.dumps(info, indent=2))
