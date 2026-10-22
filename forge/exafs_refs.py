"""FEFF reference runs for the EXAFS shortcut.

1. Pair references (single scattering): absorber X at the centre of an fcc cluster of scatterer Y with
   a = (a_X + a_Y) / 2, NLEG 2, RMAX 6.3 A -> amplitude, phase, Re(p) and mean free path for shells 1-5.
   X-O references come from a rock-salt-like cluster with d(X-O) from particles.M_O_DIST.
2. Multiple-scattering templates: bulk clusters of each composition / phase (ordered: ideal superstructure;
   disordered: 4 random configurations) with the absorber at the centre, NLEG 4, RMAX 6.0 A. Paths with
   nleg >= 3 are kept (degeneracy divided by the number of configurations).

Everything is tabulated on the stored k grid (0 ... 16 A^-1, step 0.05) in artifacts/data/work/feff/refs.npz.
"""
import json
import math
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .feff import FeffWork, run_feff, write_feff_inp
from .particles import M_O_DIST, distortion_matrix, sublattice_species
from .paths import FEFF_DIR, STRUCT_DIR, sha256_json

ABSORBERS = ["Pt", "Co", "Ni", "Fe", "Cu"]
SCATTERERS = ["Pt", "Co", "Ni", "Fe", "Cu", "O"]
K_STORE = np.round(np.arange(321) * 0.05, 4)
N_WORKERS = 28
N_DISORDER_CONFIGS = 4


def fcc_int_ball(radius_half):
    m = int(math.ceil(radius_half)) + 1
    g = np.arange(-m, m + 1)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    u = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    u = u[(u.sum(1) % 2) == 0]
    return u[np.linalg.norm(u, axis=1) <= radius_half]


def _tab(path):
    k = np.where(K_STORE == 0, 1e-6, K_STORE)
    out = dict(nleg=path["nleg"], degen=path["degen"], reff=path["reff"], atoms=path["atoms"],
               coords=path["coords"].tolist())
    for q in ("amp", "pha", "rep", "lam"):
        out[q] = np.interp(k, path["k"], path[q]).astype(np.float64)
    # distance of the farthest atom from the absorber (for coordination scaling of MS paths)
    out["r_far"] = float(np.max(np.linalg.norm(path["coords"][1:] - path["coords"][0], axis=1)))
    return out


def _job(job):
    """FEFF run; FEFF6's path hash can collide on near-degenerate ideal geometries (e.g. twinned clusters), in which
    case the run is repeated with a <= 0.01 A coordinate jitter and the jittered geometry is the one stored."""
    name, key, pos, sp, i0, rmax, nleg = job
    pos = np.asarray(pos, float)
    rng = np.random.default_rng(abs(hash(name)) % 2 ** 32)
    last = None
    for attempt in range(6):
        p = pos.copy()
        if attempt:
            p += rng.normal(size=p.shape) * 0.002 * attempt
            p[i0] = pos[i0]
        try:
            with FeffWork(FEFF_DIR / "tmp", name) as d:
                write_feff_inp(d, p, sp, i0, rmax=rmax, nleg=nleg, title=name)
                paths = run_feff(d)
            break
        except Exception as e:
            last, paths = e, None
    if paths is None:
        raise last
    cluster = dict(pos=(p - p[i0]).astype(np.float32), species=list(sp), absorber=int(i0), jitter=0.002 * attempt)
    return name, key, [_tab(q) for q in paths], cluster


def pair_jobs(a_end):
    jobs = []
    for X in ABSORBERS:
        for Y in SCATTERERS:
            if Y == "O":
                d = M_O_DIST[X]
                g = fcc_int_ball(2 * 6.5 / (2 * d))  # rock-salt: fcc of X plus O on octahedral sites
                # rock salt with X-O = d: cation fcc with a = 2d; anion fcc shifted by (d,0,0)
                ux = g * d  # half-units * d -> cation positions with a = 2d
                ox = (g + np.array([1, 0, 0])) * d
                pos = np.concatenate([ux, ox])
                sp = [X] * len(ux) + ["O"] * len(ox)
                r = np.linalg.norm(pos, axis=1)
                keep = r <= 6.5
                pos, sp = pos[keep], [s for s, kp in zip(sp, keep) if kp]
                i0 = int(np.argmin(np.linalg.norm(pos, axis=1)))
                jobs.append((f"pair_{X}_O", f"{X}-O", pos, sp, i0, 2.6, 2))
            else:
                a = 0.5 * (a_end[X] + a_end[Y])
                u = fcc_int_ball(2 * 7.2 / a)
                pos = u * a / 2.0
                sp = [Y] * len(u)
                i0 = int(np.argmin(np.linalg.norm(pos, axis=1)))
                sp[i0] = X
                jobs.append((f"pair_{X}_{Y}", f"{X}-{Y}", pos, sp, i0, 6.3, 2))
    return jobs


def twin_cluster_int(radius_half):
    """fcc integer coordinates with a {111} mirror (twin) plane x + y + z = 0 through the origin."""
    u = fcc_int_ball(radius_half)
    w = u.sum(1)
    keep = u[w >= 0].astype(float)
    src = u[w > 0].astype(float)
    refl = src - (2.0 * src.sum(1) / 3.0)[:, None]
    return np.concatenate([keep, refl])


N_TWIN_CONFIGS = 2


def env_jobs(ref):
    """Bulk clusters for multiple-scattering templates (fcc / ordered bulk, plus {111} twin-boundary environments
    for disordered phases, which occur in decahedra, icosahedra and twinned particles)."""
    from .structures import SYSTEMS
    phases = ref["phases"]
    jobs, env_a = [], {}
    rng = np.random.default_rng(20261022)
    for sys_name, spec in SYSTEMS.items():
        M = spec["M"]
        env_list = [("A1", None)]
        if spec["ordered"] and phases.get(f"{sys_name}|{spec['ordered']}", {}).get("available"):
            env_list.append((spec["ordered"], phases[f"{sys_name}|{spec['ordered']}"]["params"]))
        for phase, prm in env_list:
            if phase == "A1":
                a = ref["vegard"][sys_name]["a_vegard"] if M else ref["endpoints"]["Pt"]
                prm = dict(a=a, c=a, a_mean=a, alpha=90.0)
            u = fcc_int_ball(2 * 8.5 / prm["a_mean"])
            D = distortion_matrix(phase, prm, 0)
            pos = u @ D.T
            absorbers = ["Pt"] + ([M] if M else [])
            for X in absorbers:
                env_a[f"{sys_name}|{phase}|{X}"] = float(prm["a_mean"])
            n_cfg = 1 if (phase != "A1" or M is None) else N_DISORDER_CONFIGS
            for cfg in range(n_cfg):
                if phase == "A1":
                    sp = np.array(["Pt"] * len(u), dtype="<U2")
                    if M:
                        kM = int(round(spec["x"] * len(u)))
                        sp[rng.choice(len(u), size=kM, replace=False)] = M
                else:
                    sp = sublattice_species(u, phase, M, 0, rng)
                order = np.argsort(np.linalg.norm(pos, axis=1))
                for X in absorbers:
                    i0 = int(next(i for i in order if sp[i] == X))
                    jobs.append((f"env_{sys_name}_{phase}_{X}_{cfg}", f"{sys_name}|{phase}|{X}", pos - pos[i0], list(sp), i0, 6.0, 4))
            if phase != "A1":
                continue
            # twin-boundary environments: absorbers on the twin plane and 1 and 2 layers away from it
            ut = twin_cluster_int(2 * 13.5 / prm["a_mean"])
            post = ut * prm["a_mean"] / 2.0
            layer = np.round(ut.sum(1) / 2.0).astype(int)
            n_cfg_t = 1 if M is None else N_TWIN_CONFIGS
            for cfg in range(n_cfg_t):
                sp = np.array(["Pt"] * len(ut), dtype="<U2")
                if M:
                    kM = int(round(spec["x"] * len(ut)))
                    sp[rng.choice(len(ut), size=kM, replace=False)] = M
                dist = np.linalg.norm(post, axis=1)
                for X in absorbers:
                    for L in (0, 1, 2):
                        cand = np.flatnonzero((layer == L) & (sp == X))
                        i0 = int(cand[np.argmin(dist[cand])])
                        sel = np.flatnonzero(np.linalg.norm(post - post[i0], axis=1) <= 8.5)
                        j0 = int(np.flatnonzero(sel == i0)[0])
                        jobs.append((f"env_{sys_name}_A1twin_{X}_L{L}_{cfg}", f"{sys_name}|{phase}|{X}",
                                     post[sel] - post[i0], list(sp[sel]), j0, 6.0, 4))
            # Mackay icosahedra are built from radially compressed fcc tetrahedra; their environments get their own
            # templates (key "<comp>|A1ico|<X>"): absorbers drawn uniformly (by volume) from the interior of an
            # 11-shell icosahedron, so tetrahedral interiors, twin planes and 5-fold axes appear in their natural mix
            from ase.cluster import Icosahedron
            posi = Icosahedron("Cu", 11, latticeconstant=prm["a_mean"]).get_positions()
            posi -= posi.mean(0)
            di = np.linalg.norm(posi, axis=1)
            r_in = di.max() - 7.5
            for cfg in range(n_cfg_t):
                sp = np.array(["Pt"] * len(posi), dtype="<U2")
                if M:
                    kM = int(round(spec["x"] * len(posi)))
                    sp[rng.choice(len(posi), size=kM, replace=False)] = M
                for X in absorbers:
                    env_a[f"{sys_name}|A1ico|{X}"] = float(prm["a_mean"])
                    cand = np.flatnonzero((di <= r_in) & (sp == X))
                    n_pick = 20 if M else 40
                    for j, i0 in enumerate(rng.choice(cand, size=min(n_pick, len(cand)), replace=False)):
                        sel = np.flatnonzero(np.linalg.norm(posi - posi[i0], axis=1) <= 8.5)
                        j0 = int(np.flatnonzero(sel == i0)[0])
                        jobs.append((f"env_{sys_name}_A1ico_{X}_{j}_{cfg}", f"{sys_name}|A1ico|{X}",
                                     posi[sel] - posi[i0], list(sp[sel]), j0, 6.0, 4))
    return jobs, env_a


def main():
    ref = json.loads((STRUCT_DIR / "reference_phases.json").read_text())
    a_end = dict(ref["endpoints"])
    ejobs, env_a = env_jobs(ref)
    jobs = pair_jobs(a_end) + ejobs
    print(f"{len(jobs)} FEFF jobs on {N_WORKERS} workers", flush=True)
    results, keys, clusters = {}, {}, {}
    with ProcessPoolExecutor(N_WORKERS) as ex:
        for name, key, paths, cluster in ex.map(_job, jobs):
            results[name], keys[name], clusters[name] = paths, key, cluster
            print(f"  {name}: {len(paths)} paths", flush=True)

    pairs = {}
    for X in ABSORBERS:
        for Y in SCATTERERS:
            ps = [p for p in results[f"pair_{X}_{Y}"] if p["nleg"] == 2]
            ps = sorted(ps, key=lambda p: p["reff"])
            pairs[f"{X}-{Y}"] = dict(reff=np.array([p["reff"] for p in ps]),
                                     amp=np.stack([p["amp"] for p in ps]), pha=np.stack([p["pha"] for p in ps]),
                                     rep=np.stack([p["rep"] for p in ps]), lam=np.stack([p["lam"] for p in ps]),
                                     degen=np.array([p["degen"] for p in ps]))
    envs = {}
    for name, paths in results.items():
        if not name.startswith("env_"):
            continue
        key = keys[name]
        twin = "twin" in name
        envs.setdefault(key, dict(paths=[], n_cfg=0, n_fcc=0, clusters=[]))
        envs[key]["n_cfg"] += 1
        envs[key]["n_fcc"] += 0 if twin else 1
        for p in paths:
            p["twin"] = twin
        envs[key]["paths"] += paths
        envs[key]["clusters"].append(dict(clusters[name], twin=twin))
    for key, e in envs.items():
        for p in e["paths"]:
            p["degen"] = p["degen"] / e["n_cfg"]
    out = dict(k=K_STORE, pairs=pairs, envs=envs, a_end=a_end, env_a=env_a,
               provenance=dict(code="FEFF6L (xraylarch 2026.3.1 bundled)", n_jobs=len(jobs),
                               inputs_sha256=sha256_json([(j[0], np.round(j[2], 4).tolist(), j[3]) for j in jobs])))
    with open(FEFF_DIR / "refs.pkl", "wb") as fh:
        pickle.dump(out, fh)
    print("pairs:", {k: len(v["reff"]) for k, v in pairs.items()})
    print("envs:", {k: (v["n_cfg"], v["n_fcc"], len(v["paths"])) for k, v in envs.items()})


if __name__ == "__main__":
    main()
