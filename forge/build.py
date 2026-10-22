"""Build pipeline: plan -> particles (CPU pool) -> simulation (4-GPU work queue) -> HDF5 shards + manifest + splits.

    python -m forge.build configs/dataset_pilot.yaml [plan|particles|simulate|assemble|all]

Every stage is resumable: outputs are written atomically per sample and existing valid outputs are skipped.
"""
import datetime
import json
import math
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

from .paths import CONFIGS, FEFF_DIR, ROOT, STRUCT_DIR, load_yaml, sha256_file, sha256_json

ELEMENTS = ["Pt", "Co", "Ni", "Fe", "Cu", "O"]
ABSORBERS = ["Pt", "Co", "Ni", "Fe", "Cu"]
COMPOSITIONS = ["Pt", "Pt3Co", "PtCo", "Pt3Ni", "PtNi", "Pt3Fe", "PtCu"]
SHAPE_LIST = ["cuboctahedron", "truncated_octahedron", "sphere", "icosahedron", "decahedron"]
PHASES = ["A1", "L1_2", "L1_0", "L1_1"]
FAULTS = ["pt_oxide_shell", "pt_skin", "mixed_oxidation", "low_dopant", "multi_domain", "wrong_phase_candidate",
          "mislabeled_metadata"]


def u_dict(spec):
    u_pt = spec["u_iso_pt"]
    U = {el: u_pt * spec["u_ratio_m"] for el in ("Co", "Ni", "Fe", "Cu")}
    U["Pt"] = u_pt
    U["O"] = spec["u_iso_o"]
    return U


def _npz_bytes(**arrays):
    import io
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def _atomic_write(path, data):
    """Write bytes to a uniquely named temporary file and rename; retry transient Windows file errors."""
    import uuid
    path = Path(path).resolve()
    for attempt in range(8):
        tmp = path.with_name(f".{path.stem}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, path)
            return
        except OSError:
            if attempt == 7:
                raise
            time.sleep(0.5 * (attempt + 1))


def _atomic_savez(path, **arrays):
    _atomic_write(path, _npz_bytes(**arrays))


def _load_ok(path):
    try:
        with np.load(path, allow_pickle=False) as z:
            _ = z.files
        return True
    except Exception:
        return False


# ------------------------------------------------------------------------------------------------ particles
_G = {}


def _init_particles():
    from .particles import shape_tables
    _G["ref"] = json.loads((STRUCT_DIR / "reference_phases.json").read_text())
    _G["tables"] = shape_tables()


def particle_task(args):
    spec, out = args
    out = Path(out)
    if out.exists() and _load_ok(out):
        return spec["sample_id"], "skip", 0.0, None
    from .particles import build_particle, coordination_shells
    from .plan import phase_params
    t0 = time.perf_counter()
    rng = np.random.default_rng(spec["seed"])
    prm = phase_params(_G["ref"], spec["composition"], spec["phase"])
    p = build_particle(spec, prm, _G["tables"], rng)
    U = u_dict(spec)
    shells = coordination_shells(p.positions, p.species, p.meta["lattice_parameter"], U, spec["delta2"],
                                 spec["corr_floor"])
    meta = dict(p.meta, shells=shells, U=U)
    codes = np.array([ELEMENTS.index(s) for s in p.species], np.uint8)
    data = _npz_bytes(positions=p.positions.astype(np.float32), species=codes,
                      meta=np.frombuffer(json.dumps(meta, default=float).encode(), np.uint8))
    return spec["sample_id"], "ok", time.perf_counter() - t0, data


def stage_particles(cfg, plan):
    wd = Path(cfg["work_dir"]) / "particles"
    wd.mkdir(parents=True, exist_ok=True)
    jobs = [(s, str(wd / f"{s['sample_id']}.npz")) for s in plan]
    t0 = time.time()
    n_ok = n_skip = 0
    with ProcessPoolExecutor(cfg["cpu_workers"], initializer=_init_particles) as ex:
        for i, (sid, st, dt, data) in enumerate(ex.map(particle_task, jobs, chunksize=4)):
            if data is not None:
                _atomic_write(wd / f"{sid}.npz", data)
            n_ok += st == "ok"
            n_skip += st == "skip"
            if (i + 1) % 500 == 0:
                print(f"  particles {i + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"particles: built {n_ok}, skipped {n_skip} in {time.time() - t0:.0f}s", flush=True)


def load_particle(path):
    with np.load(path) as z:
        pos = z["positions"]
        sp = np.array(ELEMENTS, dtype="<U2")[z["species"]]
        meta = json.loads(bytes(z["meta"]).decode())
    return pos, sp, meta


# ------------------------------------------------------------------------------------------------ simulation
def mu_params(spec, X, meta, inst, rng):
    ex = inst["exafs"]
    lo, hi = ex["white_line"][X]
    ox = meta["oxidation_states"].get(X, {"0": 1.0})
    mean_ox = sum(float(k) * v for k, v in ox.items())
    return dict(edge_width=spec["instruments"]["exafs"]["edge_width"],
                white_line=float(rng.uniform(lo, hi)) + ex["white_line_per_ox"][X] * mean_ox,
                wl_pos=float(rng.uniform(2.0, 6.0)), wl_width=float(rng.uniform(4.0, 8.0)),
                shift=ex["edge_shift_per_ox"][X] * mean_ox, pre_slope=float(rng.uniform(-0.05, 0.05)),
                post_slope=float(rng.uniform(0.0, 0.1)), mean_oxidation=mean_ox)


def simulate_one(sim, exm, inst, spec, ppath):
    from .exafs import env_key, mu_from_chi
    from .motifs import SPECIES, finalize, motif_stats
    pos, sp, meta = load_particle(ppath)
    names = sorted(set(sp.tolist()), key=ELEMENTS.index)
    types = np.array([names.index(s) for s in sp])
    counts = np.bincount(types, minlength=len(names)).astype(float)
    U = u_dict(spec)
    d2, fl = spec["delta2"], spec["corr_floor"]
    t = {}
    t0 = time.perf_counter()
    H, nb = sim.histograms(pos, types, len(names))
    t["hist"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    xc = sim.xrd_cu(H, nb, names, counts, U, d2, fl, spec["instruments"]["xrd_cu"])
    xs, tth_s = sim.xrd_sync(H, nb, names, counts, U, d2, fl, spec["instruments"]["xrd_sync"])
    t["xrd"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    G, q_pdf, F = sim.pdf(H, nb, names, counts, U, d2, fl, spec["instruments"]["pdf"])
    t["pdf"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    rng = np.random.default_rng(spec["seed"] + 1)
    out = dict(xrd_cu=xc.astype(np.float32), xrd_sync=xs.astype(np.float32), pdf_gr=G.astype(np.float32),
               pdf_fq=F.astype(np.float32))
    mu_meta = {}
    a_s = meta["lattice_parameter"]
    for edge, X in (("pt", "Pt"), ("m", spec["M"])):
        if X is None:
            out[f"chi_{edge}"] = np.full(len(exm.k), np.nan, np.float32)
            out[f"mu_{edge}"] = np.full(len(sim.g.e_rel), np.nan, np.float32)
            continue
        partials, r = sim.absorber_partials(H, nb, names, counts, X)
        env = env_key(spec, X)
        codes = np.array([SPECIES.index(x) for x in sp])
        mst = finalize(motif_stats(pos, codes, np.flatnonzero(sp == X), a_s, device=sim.dev))
        chi = exm.chi(X, partials, r, env, a_s, U, d2, fl, inst["exafs"]["s02"][X], motifs=mst)
        mp = mu_params(spec, X, meta, inst, rng)
        mu = mu_from_chi(sim.g.e_rel, exm.k, chi, mp)
        out[f"chi_{edge}"] = chi.astype(np.float32)
        out[f"mu_{edge}"] = mu.astype(np.float32)
        mu_meta[X] = mp
    t["exafs"] = time.perf_counter() - t0
    info = dict(timing=t, mu_params=mu_meta, n_atoms=int(len(pos)), elements=names)
    return out, info


def gpu_worker(rank, gpu, mem_gib, ids, cfg_path):
    import torch
    from .exafs import ExafsModel
    from .simulate import GpuSim
    cfg = load_yaml(cfg_path)
    inst = load_yaml(CONFIGS / "instruments.yaml")
    wd = Path(cfg["work_dir"])
    plan = {s["sample_id"]: s for s in json.loads((wd / "plan.json").read_text())}
    torch.cuda.set_device(gpu)
    total = torch.cuda.get_device_properties(gpu).total_memory
    frac = min(0.95, mem_gib * 2 ** 30 / total)
    sim = GpuSim(inst, f"cuda:{gpu}", mem_fraction=frac)
    exm = ExafsModel(rmax_ss=inst["exafs"]["rmax_ss"], rmax_ms=inst["exafs"]["rmax_ms"], device=torch.device("cuda", gpu))
    (wd / "sims").mkdir(exist_ok=True)
    (wd / "locks").mkdir(exist_ok=True)
    (wd / "errors").mkdir(exist_ok=True)
    n_done, t_start = 0, time.time()
    log = open(wd / f"sim_worker_{rank}.log", "a", buffering=1)
    for sid in ids:
        out = wd / "sims" / f"{sid}.npz"
        if out.exists():
            continue
        try:
            fd = os.open(str(wd / "locks" / f"{sid}.lock"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            continue
        try:
            arrays, info = simulate_one(sim, exm, inst, plan[sid], wd / "particles" / f"{sid}.npz")
            info["gpu"] = gpu
            _atomic_savez(out, info=np.frombuffer(json.dumps(info).encode(), np.uint8), **arrays)
            n_done += 1
            if n_done % 50 == 0:
                log.write(f"{datetime.datetime.now():%H:%M:%S} gpu{gpu} done {n_done} "
                          f"({n_done / (time.time() - t_start):.2f}/s) last={info['timing']}\n")
        except Exception:
            (wd / "errors" / f"{sid}.txt").write_text(traceback.format_exc())
            log.write(f"ERROR {sid}\n")
        finally:
            torch.cuda.empty_cache() if n_done % 200 == 0 else None
    log.write(f"worker {rank} finished: {n_done} samples in {time.time() - t_start:.0f}s\n")


def stage_simulate(cfg, plan, cfg_path):
    import torch.multiprocessing as mp
    wd = Path(cfg["work_dir"])
    for d in ("locks", "errors"):
        p = wd / d
        p.mkdir(parents=True, exist_ok=True)
        for f in p.glob("*"):
            f.unlink()
    # largest particles first so the tail of the queue is short
    ids = [s["sample_id"] for s in sorted(plan, key=lambda s: -s["d_target_nm"])]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    procs, restarts = {}, {gpu: 0 for gpu in cfg["gpus"]}
    for rank, gpu in enumerate(cfg["gpus"]):
        procs[gpu] = (rank, ctx.Process(target=gpu_worker, args=(rank, gpu, cfg["gpu_mem_gib"][gpu], ids, cfg_path)))
        procs[gpu][1].start()
    while procs:
        time.sleep(5)
        for gpu, (rank, p) in list(procs.items()):
            if p.is_alive():
                continue
            p.join()
            del procs[gpu]
            done = len(list((wd / "sims").glob("*.npz")))
            if p.exitcode == 3 and restarts[gpu] < 3 and done < len(plan):
                restarts[gpu] += 1
                print(f"respawning worker for gpu {gpu} after CUDA error (restart {restarts[gpu]})", flush=True)
                np_ = ctx.Process(target=gpu_worker, args=(rank, gpu, cfg["gpu_mem_gib"][gpu], ids, cfg_path))
                np_.start()
                procs[gpu] = (rank, np_)
    n = len(list((wd / "sims").glob("*.npz")))
    errs = list((wd / "errors").glob("*.txt"))
    print(f"simulate: {n}/{len(plan)} done, {len(errs)} errors, {time.time() - t0:.0f}s", flush=True)
    return n, errs


# ------------------------------------------------------------------------------------------------ assembly
def _label_arrays(spec, meta):
    comp_true = meta["composition_true"]
    sh = meta["shells"]
    coord = np.full((len(ABSORBERS), len(ELEMENTS), 4, 3), np.nan, np.float32)
    for s in sh:
        coord[ABSORBERS.index(s["absorber"]), ELEMENTS.index(s["neighbor"]), s["shell"] - 1] = (s["N"], s["R"], s["sigma2"])
    first = np.full((2, 3), np.nan, np.float32)  # rows: Pt absorber, M absorber; cols: N(metal), R(metal mean), sigma2
    for row, X in enumerate(["Pt", spec["M"]]):
        if X is None:
            continue
        ss = [s for s in sh if s["absorber"] == X and s["shell"] == 1 and s["neighbor"] != "O"]
        if ss:
            n = sum(s["N"] for s in ss)
            first[row] = (n, sum(s["N"] * s["R"] for s in ss) / n, sum(s["N"] * s["sigma2"] for s in ss) / n)
    return coord, first


def stage_assemble(cfg, plan, final=False):
    wd = Path(cfg["work_dir"])
    out_dir = Path(cfg["out_dir"])
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    inst = load_yaml(CONFIGS / "instruments.yaml")
    from .simulate import Grids
    g = Grids(inst)
    ref = json.loads((STRUCT_DIR / "reference_phases.json").read_text())
    groups = defaultdict(list)
    for s in plan:
        groups[(s["set"], s["split"])].append(s)
    shard_index, sample_rows = [], []
    for (st, split), specs in sorted(groups.items()):
        specs = sorted(specs, key=lambda s: s["sample_id"])
        for i0 in range(0, len(specs), cfg["shard_size"]):
            chunk = specs[i0:i0 + cfg["shard_size"]]
            name = f"{cfg['name']}_{st}_{split}_{i0 // cfg['shard_size']:03d}.h5"
            path = shard_dir / name
            tmp = path.with_suffix(".tmp")
            arrs = defaultdict(list)
            metas = []
            for s in chunk:
                sid = s["sample_id"]
                z = np.load(wd / "sims" / f"{sid}.npz")
                pos, sp, pmeta = load_particle(wd / "particles" / f"{sid}.npz")
                info = json.loads(bytes(z["info"]).decode())
                for k in ("xrd_cu", "xrd_sync", "pdf_gr", "pdf_fq", "chi_pt", "chi_m", "mu_pt", "mu_m"):
                    arrs[k].append(z[k])
                coord, first = _label_arrays(s, pmeta)
                arrs["coordination"].append(coord)
                arrs["first_shell"].append(first)
                arrs["composition_id"].append(COMPOSITIONS.index(s["composition"]))
                arrs["phase_id"].append(PHASES.index(s["phase"]))
                arrs["ordered"].append(-1 if s["M"] is None else int(s["phase"] != "A1"))
                arrs["shape_id"].append(SHAPE_LIST.index(s["shape"]))
                arrs["size_bin"].append(s["size_bin"])
                for k in ("diameter_nm", "diameter_rg_nm", "lattice_parameter", "lattice_a", "lattice_c", "lattice_alpha",
                          "n_atoms", "n_surface"):
                    arrs[k].append(pmeta[k])
                arrs["strain"].append(s["strain"])
                arrs["vacancy_fraction"].append(s["vacancy_fraction"])
                arrs["u_iso_pt"].append(s["u_iso_pt"])
                arrs["delta2"].append(s["delta2"])
                arrs["x_M_true"].append(sum(v for k, v in pmeta["composition_true"].items() if k not in ("Pt", "O")))
                arrs["faults"].append([int(s["fault"] == f) for f in FAULTS])
                ins = s["instruments"]
                arrs["inst_xrd_cu"].append([ins["xrd_cu"]["ka2_ratio"], float(ins["xrd_cu"]["monochromator"] == "graphite"),
                                            ins["xrd_cu"]["caglioti_U"], ins["xrd_cu"]["caglioti_V"],
                                            ins["xrd_cu"]["caglioti_W"], ins["xrd_cu"]["lorentz_fraction"],
                                            ins["xrd_cu"]["counts_max"]])
                arrs["inst_xrd_sync"].append([ins["xrd_sync"]["wavelength"], ins["xrd_sync"]["fwhm_q0"],
                                              ins["xrd_sync"]["fwhm_q1"], ins["xrd_sync"]["counts_max"]])
                arrs["inst_pdf"].append([ins["pdf"]["qmin"], ins["pdf"]["qmax"], ins["pdf"]["qdamp"],
                                         ins["pdf"]["qbroad"], ins["pdf"]["noise_rel"]])
                arrs["inst_exafs"].append([ins["exafs"]["noise_chi"], ins["exafs"]["edge_width"]])
                arrs["sim_sha256"].append(sha256_file(wd / "sims" / f"{sid}.npz"))
                rep = ref["phases"].get(f"{s['composition']}|{s['phase']}", {}).get("representative")
                full = dict(spec=s, particle=pmeta, sim=info, structure_source=rep)
                metas.append(json.dumps(full, default=float))
                sample_rows.append(dict(sample_id=sid, set=st, split=split, family=s["family"], shard=name,
                                        composition=s["composition"], phase=s["phase"], shape=s["shape"],
                                        size_bin=s["size_bin"], fault=s["fault"] or "",
                                        structure_source=(rep or {"source": "Vegard (COD endpoints)"})["source"],
                                        structure_id=(rep or {}).get("source_id", "vegard"),
                                        sim_sha256=arrs["sim_sha256"][-1]))
            with h5py.File(tmp, "w") as h:
                h.attrs["dataset"] = "Concord Forge Pt/Pt-alloy nanoparticles"
                h.attrs["build"] = cfg["name"]
                h.attrs["set"] = st
                h.attrs["split"] = split
                h.attrs["origin"] = "simulated"
                h.attrs["created"] = datetime.datetime.now().isoformat(timespec="seconds")
                h.attrs["compositions"] = json.dumps(COMPOSITIONS)
                h.attrs["phases"] = json.dumps(PHASES)
                h.attrs["shapes"] = json.dumps(SHAPE_LIST)
                h.attrs["faults"] = json.dumps(FAULTS)
                h.attrs["absorbers"] = json.dumps(ABSORBERS)
                h.attrs["neighbors"] = json.dumps(ELEMENTS)
                h.create_dataset("grid/xrd_cu_two_theta", data=g.tth.astype(np.float32))
                h.create_dataset("grid/xrd_sync_q", data=g.q_sync.astype(np.float32))
                h.create_dataset("grid/pdf_r", data=g.r_pdf.astype(np.float32))
                h.create_dataset("grid/pdf_q", data=g.q_pdf.astype(np.float32))
                h.create_dataset("grid/exafs_k", data=np.asarray(np.arange(321) * 0.05, np.float32))
                h.create_dataset("grid/mu_e_rel", data=g.e_rel.astype(np.float32))
                h.create_dataset("sample_id", data=np.array([s["sample_id"] for s in chunk], dtype="S16"))
                for k, v in arrs.items():
                    if k == "sim_sha256":
                        h.create_dataset(k, data=np.array(v, dtype="S64"))
                    else:
                        a = np.asarray(v)
                        h.create_dataset(k, data=a.astype(np.float32) if a.dtype.kind == "f" else a)
                h.create_dataset("meta_json", data=np.array(metas, dtype=h5py.string_dtype()))
            os.replace(tmp, path)
            shard_index.append(dict(shard=name, set=st, split=split, n=len(chunk), sha256=sha256_file(path)))
            print(f"  shard {name}: {len(chunk)} samples", flush=True)
    return shard_index, sample_rows


def write_manifest(cfg, plan, shard_index, sample_rows):
    import csv
    out_dir = Path(cfg["out_dir"])
    ref = json.loads((STRUCT_DIR / "reference_phases.json").read_text())
    lic = {"COD": "CC0-1.0", "JARVIS": "CC-BY-4.0", "OQMD": "CC-BY-4.0", "MP": "CC-BY-4.0"}
    rows = []
    for key, ph in ref["phases"].items():
        if not ph.get("available"):
            rows.append(dict(kind="structure", item_id=key, source="none", source_id="", licence="",
                             notes="no open structure found - phase skipped"))
            continue
        used = [m for m in ph["members"] if m.get("used", True)]
        for m in ph["members"]:
            rows.append(dict(kind="structure", item_id=key, source=m["source"], source_id=m["source_id"],
                             licence=lic[m["source"]], notes=("used" if m.get("used") else "duplicate (not used)")
                             + (" kX->A" if m.get("kx_converted") else "") + f"; basis: {ph['basis']}"))
    rows.append(dict(kind="feff_reference", item_id="refs.pkl", source="FEFF6L (xraylarch 2026.3.1)",
                     source_id=sha256_file(FEFF_DIR / "refs.pkl"), licence="derived (computed here)",
                     notes="pair and bulk multiple-scattering reference runs"))
    for r in sample_rows:
        src = r["structure_source"]
        rows.append(dict(kind="sample", item_id=r["sample_id"], source=f"simulated from {src}",
                         source_id=r["structure_id"], licence="CC-BY-4.0" if src in ("JARVIS", "OQMD") else "CC0-1.0",
                         notes=f"set={r['set']};split={r['split']};family={r['family']};shard={r['shard']};"
                               f"fault={r['fault']};sha256={r['sim_sha256']}"))
    for s in shard_index:
        rows.append(dict(kind="shard", item_id=s["shard"], source="simulated", source_id=s["sha256"],
                         licence="CC-BY-4.0", notes=f"set={s['set']};split={s['split']};n={s['n']}"))
    real = ROOT / "artifacts" / "data" / "real_raw" / "catalog.csv"
    if real.exists():
        with open(real, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                rows.append(dict(kind="real_spectrum", item_id=r.get("item_id", ""), source=r.get("source", ""),
                                 source_id=r.get("source_record_url", ""), licence=r.get("licence", ""),
                                 notes=f"{r.get('technique', '')};{r.get('edge_or_wavelength', '')};{r.get('material', '')}"))
    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["kind", "item_id", "source", "source_id", "licence", "notes"])
        w.writeheader()
        w.writerows(rows)
    splits = dict(train=[], val=[], test=[], fault=dict(train=[], val=[], test=[]), families=defaultdict(list))
    for s in sorted(plan, key=lambda s: s["sample_id"]):
        if s["set"] == "clean":
            splits[s["split"]].append(s["sample_id"])
        else:
            splits["fault"][s["split"]].append(s["sample_id"])
    fam = defaultdict(set)
    for s in plan:
        fam[s["split"]].add(s["family"])
    splits["families"] = {k: sorted(v) for k, v in fam.items()}
    splits["shards"] = shard_index
    (out_dir / "splits.json").write_text(json.dumps(splits, indent=1))
    print(f"manifest: {len(rows)} rows; splits: " + ", ".join(
        f"{k}={len(splits[k])}" for k in ("train", "val", "test")) + ", fault=" + str(
        {k: len(v) for k, v in splits["fault"].items()}), flush=True)


def main():
    cfg_path = sys.argv[1]
    stage = sys.argv[2] if len(sys.argv) > 2 else "all"
    cfg = load_yaml(cfg_path)
    wd = Path(cfg["work_dir"])
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "config_used.json").write_text(json.dumps(dict(cfg=cfg, instruments=load_yaml(CONFIGS / "instruments.yaml")), indent=1))
    if stage in ("plan", "all") or not (wd / "plan.json").exists():
        from .plan import main as plan_main
        plan_main(cfg_path)
    plan = json.loads((wd / "plan.json").read_text())
    if stage in ("particles", "all"):
        stage_particles(cfg, plan)
    if stage in ("simulate", "all"):
        stage_simulate(cfg, plan, cfg_path)
    if stage == "worker":
        # extra worker joining a running simulate stage (does not clear the lock directory)
        gpu, rank = int(sys.argv[3]), int(sys.argv[4])
        ids = [s["sample_id"] for s in sorted(plan, key=lambda s: -s["d_target_nm"])]
        gpu_worker(rank, gpu, float(sys.argv[5]) if len(sys.argv) > 5 else cfg["gpu_mem_gib"][gpu], ids, cfg_path)
    if stage in ("assemble", "all"):
        shard_index, rows = stage_assemble(cfg, plan)
        write_manifest(cfg, plan, shard_index, rows)


if __name__ == "__main__":
    main()
