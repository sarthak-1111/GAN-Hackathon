#!/usr/bin/env python
"""Validate a Concord Forge dataset build. Exits 0 only if every check passes.

    python artifacts/scripts/validate_dataset.py                      # artifacts/data if built, else the pilot
    python artifacts/scripts/validate_dataset.py --data artifacts/data/pilot

Checks
  files      manifest.csv, splits.json and every shard exist; shard SHA-256 matches splits.json; per-sample simulation
             hashes in the shards match manifest.csv
  schema     every shard has the expected datasets, shapes and dtypes; no NaN / Inf except where a field does not apply
             (M-edge EXAFS of pure Pt, unused coordination slots)
  balance    15 % planted faults; equal compositions; ordered = disordered within each alloy; equal shapes within a
             phase; equal size bins within a shape (tolerances below)
  splits     no sample in two splits; no structure family in two splits; split fractions close to 80 / 10 / 10 for the
             clean set and for the fault set
  labels     value ranges (lattice parameter, size, strain, vacancies, coordination numbers, composition)
  physics    label <-> spectrum consistency: Cu K-alpha (111) peak vs lattice parameter, PDF first peak vs first-shell R,
             EXAFS first-shell Fourier peak present, mu(E) edge step ~ 1
  licences   every structure, sample and real spectrum row in manifest.csv has a licence
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COMPOSITIONS = ["Pt", "Pt3Co", "PtCo", "Pt3Ni", "PtNi", "Pt3Fe", "PtCu"]
SPECTRA = {"xrd_cu": 2048, "xrd_sync": 2048, "pdf_gr": 3072, "pdf_fq": None, "chi_pt": 321, "chi_m": 321,
           "mu_pt": None, "mu_m": None}
LABELS = ["composition_id", "phase_id", "ordered", "shape_id", "size_bin", "diameter_nm", "lattice_parameter", "strain",
          "vacancy_fraction", "faults", "coordination", "first_shell", "x_M_true", "sample_id", "meta_json"]
TOL = dict(fault_fraction=0.005, comp_rel=0.03, phase_rel=0.04, shape_rel=0.06, bin_rel=0.08, split_abs=0.03,
           fault_split_abs=0.05, phys_pass=0.97)
FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel_spread(counts):
    v = np.array(list(counts.values()), float)
    return float((v.max() - v.min()) / v.mean()) if len(v) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    args = ap.parse_args()
    data = Path(args.data) if args.data else (ROOT / "artifacts" / "data" if (ROOT / "artifacts" / "data" / "splits.json").exists()
                                              else ROOT / "artifacts" / "data" / "pilot")
    print(f"validating {data}")
    man_p, spl_p = data / "manifest.csv", data / "splits.json"
    check("files: manifest.csv and splits.json exist", man_p.exists() and spl_p.exists())
    if FAILS:
        return 1
    splits = json.loads(spl_p.read_text())
    manifest = list(csv.DictReader(open(man_p, encoding="utf-8")))
    shards = splits["shards"]
    missing = [s["shard"] for s in shards if not (data / "shards" / s["shard"]).exists()]
    check("files: all shards present", not missing, f"{len(shards)} shards, missing {missing[:3]}")
    bad_hash = [s["shard"] for s in shards if not missing and sha256(data / "shards" / s["shard"]) != s["sha256"]]
    check("files: shard SHA-256 match splits.json", not bad_hash, f"mismatch {bad_hash[:3]}")
    if missing:
        return 1

    # ------------------------------------------------------------------ load labels / spectra summaries
    rows = []
    phys = defaultdict(list)
    schema_err, nan_err = [], []
    sim_hash = {}
    for s in shards:
        with h5py.File(data / "shards" / s["shard"], "r") as h:
            n = h["sample_id"].shape[0]
            for k, L in SPECTRA.items():
                if k not in h or h[k].shape[0] != n or (L and h[k].shape[1] != L):
                    schema_err.append(f"{s['shard']}:{k}")
            for k in LABELS:
                if k not in h or h[k].shape[0] != n:
                    schema_err.append(f"{s['shard']}:{k}")
            if schema_err:
                continue
            comp = h["composition_id"][:]
            pure = comp == 0
            for k in SPECTRA:
                a = h[k][:]
                fin = np.isfinite(a)
                if k in ("chi_m", "mu_m"):
                    if not (fin[~pure].all() and (~fin[pure]).all()):
                        nan_err.append(f"{s['shard']}:{k}")
                elif not fin.all():
                    nan_err.append(f"{s['shard']}:{k}")
            ids = [x.decode() for x in h["sample_id"][:]]
            for i, sid in enumerate(ids):
                sim_hash[sid] = h["sim_sha256"][i].decode()
            tth = h["grid/xrd_cu_two_theta"][:]
            r = h["grid/pdf_r"][:]
            k_ex = h["grid/exafs_k"][:]
            e_rel = h["grid/mu_e_rel"][:]
            lat = h["lattice_parameter"][:]
            size = h["diameter_nm"][:]
            shape = h["shape_id"][:]
            first = h["first_shell"][:]
            xc = h["xrd_cu"][:]
            G = h["pdf_gr"][:]
            chi = h["chi_pt"][:]
            mu = h["mu_pt"][:]
            faults = h["faults"][:]
            for i, sid in enumerate(ids):
                meta = json.loads(h["meta_json"][i])
                sp = meta["spec"]
                rows.append(dict(sample_id=sid, set=s["set"], split=s["split"], family=sp["family"],
                                 composition=COMPOSITIONS[comp[i]], phase=sp["phase"], shape=sp["shape"],
                                 size_bin=sp["size_bin"], fault=sp["fault"] or "", lattice=float(lat[i]),
                                 size=float(size[i]), strain=float(h["strain"][i]), vac=float(h["vacancy_fraction"][i]),
                                 xM=float(h["x_M_true"][i]), first=first[i]))
                physical_fault = faults[i, :5].any()
                # Cu K-alpha (111): strongest maximum in 35-45 deg vs Bragg angle (single crystals >= 3 nm, no faults)
                if size[i] >= 3.0 and shape[i] in (0, 1, 2) and not physical_fault:
                    m = (tth > 35) & (tth < 45)
                    tth_obs = tth[m][np.argmax(xc[i, m])]
                    tth_exp = 2 * math.degrees(math.asin(1.540562 * math.sqrt(3) / (2 * lat[i])))
                    phys["xrd111"].append(abs(tth_obs - tth_exp) < 0.5)
                # PDF first peak vs label first-shell R of Pt (metal neighbours)
                if not physical_fault and np.isfinite(first[i, 0, 1]):
                    m = (r > 2.2) & (r < 3.2)
                    r_obs = r[m][np.argmax(G[i, m])]
                    phys["pdf_r1"].append(abs(r_obs - first[i, 0, 1]) < 0.05)
                # EXAFS: k^2 chi Fourier magnitude peaks in the first-shell region
                kk = (k_ex >= 3) & (k_ex <= 12)
                rg = np.linspace(0.5, 6.0, 111)
                y = chi[i, kk] * k_ex[kk] ** 2 * np.hanning(kk.sum())
                ft = np.abs((y[None, :] * np.exp(2j * k_ex[kk][None, :] * rg[:, None])).sum(1))
                phys["exafs_r1"].append(1.6 <= rg[np.argmax(ft)] <= 2.9)
                # mu(E): edge step ~ 1 at +150 eV
                j = np.argmin(np.abs(e_rel - 150))
                phys["mu_step"].append(0.75 < mu[i, j] < 1.25)
    check("schema: datasets, shapes", not schema_err, f"errors {schema_err[:4]}")
    check("schema: finite values (M edge NaN only for pure Pt)", not nan_err, f"errors {nan_err[:4]}")
    if schema_err:
        return 1

    n = len(rows)
    n_split = len(splits["train"]) + len(splits["val"]) + len(splits["test"]) + sum(len(v) for v in splits["fault"].values())
    check("counts: shards == splits.json", n == n_split, f"{n} in shards, {n_split} in splits.json")
    ids = [r["sample_id"] for r in rows]
    check("counts: unique sample ids", len(set(ids)) == n)

    # ------------------------------------------------------------------ balance
    nf = sum(1 for r in rows if r["fault"])
    check("balance: planted-fault fraction 15 %", abs(nf / n - 0.15) <= TOL["fault_fraction"], f"{nf}/{n} = {nf / n:.4f}")
    check("balance: fault types equal", rel_spread(Counter(r["fault"] for r in rows if r["fault"])) <= 0.1,
          str(dict(Counter(r["fault"] for r in rows if r["fault"]))))
    clean = [r for r in rows if not r["fault"]]
    cc = Counter(r["composition"] for r in clean)
    check("balance: compositions equal (clean)", rel_spread(cc) <= TOL["comp_rel"], str(dict(cc)))
    worst = 0.0
    for comp in COMPOSITIONS[1:]:
        pc = Counter(r["phase"] == "A1" for r in clean if r["composition"] == comp)
        worst = max(worst, rel_spread(pc))
    check("balance: ordered = disordered within each alloy", worst <= TOL["phase_rel"], f"max spread {worst:.3f}")
    worst = 0.0
    for key in {(r["composition"], r["phase"]) for r in clean}:
        worst = max(worst, rel_spread(Counter(r["shape"] for r in clean if (r["composition"], r["phase"]) == key)))
    check("balance: shapes equal within composition/phase", worst <= TOL["shape_rel"], f"max spread {worst:.3f}")
    bc = Counter(r["size_bin"] for r in clean)
    check("balance: size bins equal (clean)", rel_spread(bc) <= TOL["bin_rel"], str(dict(bc)))

    # ------------------------------------------------------------------ splits
    sets = {k: set(splits[k]) for k in ("train", "val", "test")}
    fsets = {k: set(v) for k, v in splits["fault"].items()}
    allsets = list(sets.values()) + list(fsets.values())
    overlap = sum(len(a & b) for i, a in enumerate(allsets) for b in allsets[i + 1:])
    check("splits: sample ids disjoint", overlap == 0, f"{overlap} overlaps")
    fam_split = defaultdict(set)
    for r in rows:
        fam_split[r["family"]].add(r["split"])
    crossing = [f for f, v in fam_split.items() if len(v) > 1]
    check("splits: structure families never cross splits", not crossing, f"{len(fam_split)} families, {len(crossing)} crossing")
    nc = len(clean)
    fr = {k: len(sets[k]) / nc for k in sets}
    check("splits: clean 80/10/10", all(abs(fr[k] - t) <= TOL["split_abs"] for k, t in (("train", .8), ("val", .1), ("test", .1))),
          ", ".join(f"{k}={v:.3f}" for k, v in fr.items()))
    ffr = {k: len(v) / max(nf, 1) for k, v in fsets.items()}
    check("splits: fault 80/10/10", all(abs(ffr[k] - t) <= TOL["fault_split_abs"] for k, t in (("train", .8), ("val", .1), ("test", .1))),
          ", ".join(f"{k}={v:.3f}" for k, v in ffr.items()))
    mism = [r["sample_id"] for r in rows if r["sample_id"] not in (sets[r["split"]] if not r["fault"] else fsets[r["split"]])]
    check("splits: shard split labels agree with splits.json", not mism, f"{len(mism)} mismatches")

    # ------------------------------------------------------------------ labels
    lat = np.array([r["lattice"] for r in rows])
    size = np.array([r["size"] for r in rows])
    first = np.array([r["first"] for r in rows])
    check("labels: lattice parameter 3.40-4.05 A", bool(((lat > 3.40) & (lat < 4.05)).all()), f"{lat.min():.3f}-{lat.max():.3f}")
    check("labels: size 1.2-9.0 nm", bool(((size > 1.2) & (size < 9.0)).all()), f"{size.min():.2f}-{size.max():.2f}")
    st = np.array([r["strain"] for r in rows])
    va = np.array([r["vac"] for r in rows])
    check("labels: strain within +-2 %, vacancies 0-3 %", bool((np.abs(st) <= 0.02).all() and ((va >= 0) & (va <= 0.03)).all()))
    n1 = first[:, :, 0]
    ok = np.isnan(n1) | ((n1 > 2) & (n1 <= 12.0001))
    check("labels: first-shell N in (2, 12]", bool(ok.all()), f"min {np.nanmin(n1):.2f} max {np.nanmax(n1):.2f}")
    r1 = first[:, :, 1]
    # Upper bound from the construction rules: largest lattice parameter (Pt, +2 % strain) a = 4.002 A, NN a/sqrt2 =
    # 2.830 A; Mackay icosahedra have tangential bonds 1.0515 x the radial NN (ASE Icosahedron, mean 1.027-1.037 x);
    # surface relaxation up to +0.5 % -> 2.830 * 1.0515 * 1.005 = 2.99 A. (The first version used 2.9 A, which ignored
    # the icosahedral strain; the full build has 47 strained pure-Pt icosahedra with mean R1 2.900-2.915 A.)
    check("labels: first-shell R 2.3-3.0 A", bool((np.isnan(r1) | ((r1 > 2.3) & (r1 < 3.0))).all()),
          f"{np.nanmin(r1):.3f}-{np.nanmax(r1):.3f}")
    xm_bad = [r["sample_id"] for r in rows if r["composition"] != "Pt" and not r["fault"] and
              abs(r["xM"] - (0.25 if r["composition"].startswith("Pt3") else 0.5)) > 0.06]
    check("labels: alloy fraction matches composition (clean)", not xm_bad, f"{len(xm_bad)} off")

    # ------------------------------------------------------------------ physics consistency
    for k, v in phys.items():
        frac = float(np.mean(v)) if v else 0.0
        check(f"physics: {k} consistent", frac >= TOL["phys_pass"], f"{frac * 100:.1f} % of {len(v)}")

    # ------------------------------------------------------------------ manifest / licences
    man_samples = {r["item_id"]: r for r in manifest if r["kind"] == "sample"}
    miss = [i for i in ids if i not in man_samples]
    check("manifest: every sample listed", not miss, f"{len(miss)} missing")
    lic_missing = [r["item_id"] for r in manifest if r["kind"] in ("structure", "sample", "shard", "real_spectrum")
                   and r["source"] != "none" and not r["licence"].strip()]
    check("manifest: licence recorded for every item", not lic_missing, f"{len(lic_missing)} without licence")
    hash_bad = [i for i in ids if i in man_samples and f"sha256={sim_hash[i]}" not in man_samples[i]["notes"]]
    check("manifest: per-sample hashes match shards", not hash_bad, f"{len(hash_bad)} mismatches")
    used = [r for r in manifest if r["kind"] == "structure" and r["notes"].startswith("used")]
    check("manifest: structure sources recorded", len(used) >= 7, f"{len(used)} used structure entries")

    print(f"\n{n} samples, {len(shards)} shards: " + ("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
