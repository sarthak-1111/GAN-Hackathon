"""Validate the EXAFS shortcut against full FEFF6L on random samples.

For each sample and edge (Pt L3; M K for alloys) up to N_ABS absorbers are drawn at random. Each absorber's
cluster (atoms within 7.5 A) is run through FEFF6L (RMAX 6.0, NLEG 4, CRITERIA 0 0 = no path filter); chi(k) is summed over
paths with sigma^2 = (U_X + mean U of the path's scatterers) * max(floor, 1 - delta2 / reff^2) - the same thermal
model the shortcut uses - and averaged over the absorbers. The shortcut is evaluated on exactly the same absorbers
and geometry (absorber-centred partial distributions of those absorbers only). S0^2 = 1 for both.

Tolerances were fixed before the first run (artifacts/reports/exafs_shortcut_validation.md):
  * R-factor of k^2 chi(k), k = 3-12 A^-1: median <= 0.02 and 95th percentile <= 0.05
  * first-shell |chi(R)| peak (k^2, Hanning 3-12 A^-1): median |relative error| <= 5 %, median |dR| <= 0.02 A
"""
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .build import load_particle, u_dict
from .exafs import ExafsModel, env_key
from .feff import FeffWork, chi_from_paths, run_feff, write_feff_inp
from .motifs import SPECIES, finalize, merge, motif_stats
from .paths import CONFIGS, FEFF_DIR, REPORTS, WORK, load_yaml
from .simulate import DR

N_ABS = 8
N_SAMPLES = 200
CLUSTER_R = 7.5
RMAX_PARTIAL = 6.6
TOL = dict(r_median=0.02, r_p95=0.05, amp_median=0.05, dr_median=0.02)


def partial_hist(pos, sp, i0, names, rmax=RMAX_PARTIAL):
    d = np.linalg.norm(pos - pos[i0], axis=1)
    nb = int(rmax / DR) + 2
    out = {}
    for Y in names:
        m = (sp == Y) & (d > 1e-6) & (d < rmax)
        u = d[m] / DR
        b0 = np.floor(u).astype(int)
        w1 = u - b0
        h = np.zeros(nb)
        np.add.at(h, b0, 1 - w1)
        np.add.at(h, b0 + 1, w1)
        out[Y] = h[: int(rmax / DR)]
    return out


def absorber_job(args):
    sid, X, i0, pos, sp, U, delta2, floor, k, a_latt = args
    d = np.linalg.norm(pos - pos[i0], axis=1)
    idx = np.flatnonzero(d <= CLUSTER_R)
    a0 = int(np.flatnonzero(idx == i0)[0])
    cpos, csp = pos[idx].astype(float), sp[idx]
    rng = np.random.default_rng(abs(hash((sid, int(i0)))) % 2 ** 32)
    last = None
    for attempt in range(6):
        jit = 0.0 if attempt == 0 else 0.002 * attempt
        p = cpos + (rng.normal(size=cpos.shape) * jit if jit else 0.0)
        if jit:
            p[a0] = cpos[a0]
        try:
            with FeffWork(FEFF_DIR / "tmp", f"val_{sid}_{X}_{i0}") as fd:
                write_feff_inp(fd, p, list(csp), a0, rmax=6.0, nleg=4, title=f"{sid} {X} {i0}")
                paths = run_feff(fd)
            break
        except Exception as e:
            last = e
            paths = None
    if paths is None:
        return dict(sid=sid, X=X, i0=int(i0), ok=False, error=str(last)[-300:])

    def s2(path):
        sc = [a for a in path["atoms"][1:]]
        us = np.mean([U.get(a, U[X]) for a in sc])
        return (U[X] + us) * max(floor, 1.0 - delta2 / path["reff"] ** 2)

    chi_ss = chi_from_paths([q for q in paths if q["nleg"] == 2], k, s2, s02=1.0)
    chi_ms = chi_from_paths([q for q in paths if q["nleg"] > 2], k, s2, s02=1.0)
    # the geometry actually used by FEFF is kept so the shortcut can be re-evaluated on it without re-running FEFF
    return dict(sid=sid, X=X, i0=int(i0), ok=True, chi=chi_ss + chi_ms, chi_ss=chi_ss, chi_ms=chi_ms,
                pos=p.astype(np.float32), species=list(csp), a0=a0, a_latt=a_latt, n_paths=len(paths), jitter=jit)


def ft_mag(k, chi, rgrid, kmin=3.0, kmax=12.0, w=2):
    m = (k >= kmin) & (k <= kmax)
    win = np.hanning(m.sum())
    kk = k[m]
    y = chi[m] * kk ** w * win
    return np.abs((y[None, :] * np.exp(2j * kk[None, :] * rgrid[:, None])).sum(1)) * (k[1] - k[0]) / np.sqrt(np.pi)


def metrics(k, chi_s, chi_f):
    m = (k >= 3) & (k <= 12)
    y_s, y_f = chi_s[m] * k[m] ** 2, chi_f[m] * k[m] ** 2
    R = float(((y_s - y_f) ** 2).sum() / (y_f ** 2).sum())
    rg = np.linspace(1.0, 3.2, 441)
    fs, ff = ft_mag(k, chi_s, rg), ft_mag(k, chi_f, rg)
    i_s, i_f = int(np.argmax(fs)), int(np.argmax(ff))
    return dict(R=R, amp_rel=float((fs[i_s] - ff[i_f]) / ff[i_f]), dR=float(rg[i_s] - rg[i_f]))


def run_feff_cached(plan_path, n_samples, seed, cache_name, ids=None, workers=28):
    """FEFF for the randomly drawn samples / absorbers (or the given sample ids); results and geometries cached in
    artifacts/data/work."""
    cache = WORK / f"exafs_val_{cache_name}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    inst = load_yaml(CONFIGS / "instruments.yaml")
    wd = Path(plan_path).parent
    plan = json.loads(Path(plan_path).read_text())
    rng = np.random.default_rng(seed)
    if ids is not None:
        byid = {s["sample_id"]: s for s in plan}
        picks = [byid[i] for i in ids]
    else:
        picks = [plan[i] for i in rng.choice(len(plan), size=min(n_samples, len(plan)), replace=False)]
    k = np.round(np.arange(321) * 0.05, 4)
    jobs, meta = [], {}
    for s in picks:
        pos, sp, pm = load_particle(wd / "particles" / f"{s['sample_id']}.npz")
        U = u_dict(s)
        meta[s["sample_id"]] = (s, pm)
        for X in ["Pt"] + ([s["M"]] if s["M"] else []):
            cand = np.flatnonzero(sp == X)
            for i0 in rng.choice(cand, size=min(N_ABS, len(cand)), replace=False):
                jobs.append((s["sample_id"], X, int(i0), pos, sp, U, s["delta2"], s["corr_floor"], k,
                             pm["lattice_parameter"]))
    print(f"{len(jobs)} FEFF absorber runs for {len(picks)} samples", flush=True)
    t0 = time.time()
    res = []
    with ProcessPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(absorber_job, jobs, chunksize=2)):
            res.append(r)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    out = dict(res=res, meta=meta, k=k, n_jobs=len(jobs), seconds=time.time() - t0, seed=seed, n_samples=len(picks))
    cache.write_bytes(pickle.dumps(out))
    return out


def evaluate(cached, out_name):
    """Shortcut vs cached full-FEFF results; writes artifacts/reports/<out_name>.json and returns the summary."""
    inst = load_yaml(CONFIGS / "instruments.yaml")
    exm = ExafsModel(rmax_ss=inst["exafs"]["rmax_ss"], rmax_ms=inst["exafs"]["rmax_ms"])
    k = exm.k
    res, meta = cached["res"], cached["meta"]
    failed = [r for r in res if not r["ok"]]
    groups = {}
    for r in res:
        if r["ok"]:
            groups.setdefault((r["sid"], r["X"]), []).append(r)
    rows = []
    r_grid = np.arange(int(RMAX_PARTIAL / DR)) * DR
    for (sid, X), rs in groups.items():
        s, pm = meta[sid]
        U = u_dict(s)
        chi_f = np.mean([r["chi"] for r in rs], axis=0)
        chi_f_ss = np.mean([r["chi_ss"] for r in rs], axis=0)
        parts, msts = [], []
        for r in rs:
            sp = np.asarray(r["species"])
            parts.append(partial_hist(r["pos"].astype(float), sp, r["a0"], sorted(set(r["species"]))))
            msts.append(motif_stats(r["pos"], np.array([SPECIES.index(x) for x in r["species"]]), [r["a0"]], r["a_latt"]))
        names = sorted({Y for p in parts for Y in p})
        part = {Y: np.mean([p.get(Y, np.zeros_like(r_grid)) for p in parts], axis=0) for Y in names}
        env = env_key(s, X)
        motifs = finalize(merge(msts))
        chi_s = exm.chi(X, part, r_grid, env, pm["lattice_parameter"], U, s["delta2"], s["corr_floor"], 1.0,
                        motifs=motifs)
        chi_ss = exm.chi(X, part, r_grid, env, pm["lattice_parameter"], U, s["delta2"], s["corr_floor"], 1.0,
                         include_ms=False)
        m = metrics(k, chi_s, chi_f)
        m_ss = metrics(k, chi_ss, chi_f)
        kk = (k >= 3) & (k <= 12)
        den = ((chi_f[kk] * k[kk] ** 2) ** 2).sum()
        err_ss = float((((chi_ss - chi_f_ss)[kk] * k[kk] ** 2) ** 2).sum() / den)
        err_ms = float((((chi_s - chi_ss - (chi_f - chi_f_ss))[kk] * k[kk] ** 2) ** 2).sum() / den)
        rows.append(dict(sample_id=sid, edge=X, composition=s["composition"], phase=s["phase"], shape=s["shape"],
                         d_nm=pm["diameter_nm"], fault=s["fault"] or "", n_abs=len(rs),
                         n_paths_mean=float(np.mean([r["n_paths"] for r in rs])), **m,
                         R_ss_only=m_ss["R"], R_part_ss=err_ss, R_part_ms=err_ms, chi_s=chi_s.tolist(),
                         chi_f=chi_f.tolist()))
    out = dict(rows=rows, failed=[dict(sid=f["sid"], X=f["X"], i0=f["i0"], error=f["error"]) for f in failed],
               tolerances=TOL, k=k.tolist(), n_jobs=cached["n_jobs"], seconds=cached["seconds"],
               seed=cached.get("seed"), n_samples=cached.get("n_samples"))
    (REPORTS / f"{out_name}.json").write_text(json.dumps(out))
    R = np.array([r["R"] for r in rows])
    amp = np.abs([r["amp_rel"] for r in rows])
    dr = np.abs([r["dR"] for r in rows])
    summary = dict(n=len(rows), R_median=float(np.median(R)), R_p95=float(np.percentile(R, 95)),
                   amp_median=float(np.median(amp)), dR_median=float(np.median(dr)),
                   R_ss_only_median=float(np.median([r["R_ss_only"] for r in rows])), failed=len(failed))
    summary["pass"] = bool(summary["R_median"] <= TOL["r_median"] and summary["R_p95"] <= TOL["r_p95"]
                           and summary["amp_median"] <= TOL["amp_median"] and summary["dR_median"] <= TOL["dr_median"])
    print(json.dumps(summary, indent=1))
    return summary


def main(plan_path, out_name="exafs_shortcut_validation", n_samples=N_SAMPLES, seed=7, cache_name=None):
    cached = run_feff_cached(plan_path, n_samples, seed, cache_name or f"{out_name}_seed{seed}")
    return evaluate(cached, out_name)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
