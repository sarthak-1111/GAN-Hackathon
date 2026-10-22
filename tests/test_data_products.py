"""Integration checks of the data products (dataset validator, Concord contract export)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "artifacts" / "data" / "contract_export" / "samples"


def test_validate_dataset_exit_zero():
    r = subprocess.run([sys.executable, str(ROOT / "artifacts" / "scripts" / "validate_dataset.py")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-3000:]


@pytest.mark.skipif(not EXPORT.exists(), reason="contract export not built yet")
def test_contract_export_format():
    dirs = sorted(p for p in EXPORT.iterdir() if p.is_dir())
    assert dirs
    for d in dirs[:200]:
        meta = json.loads((d / "sample.json").read_text())
        for key in ("sample_id", "composition", "techniques", "instrument", "origin", "ground_truth", "faults"):
            assert key in meta, f"{d.name}: {key} missing"
        assert meta["origin"] in ("simulated", "simulated+realism")
        for f, ncol in (("xrd.xy", 3), ("pdf.gr", 3), ("xafs_mu.dat", 2), ("xafs_chi.k", 2)):
            a = np.loadtxt(d / f)
            assert a.ndim == 2 and a.shape[1] == ncol and np.isfinite(a).all(), f"{d.name}/{f}"
        for f in meta.get("extra_files", []):
            assert (d / f).exists(), f"{d.name}: listed extra file {f} missing"
