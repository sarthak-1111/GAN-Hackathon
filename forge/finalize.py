"""Final stage: GAN diagnostics, downstream comparison + decision rule, shipping and cards.

    python -m forge.finalize select           # choose between alternative realism runs (validation score only)
    python -m forge.finalize diagnostics      # reports/gan_diagnostics.md (needs trained realism models + predictor)
    python -m forge.finalize downstream       # reports/downstream_comparison.md, ships the winning predictor
    python -m forge.finalize cards            # model cards + dataset card
    python -m forge.finalize tidy             # intermediate checkpoints and unselected runs -> artifacts/models/work/
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from .paths import MODELS, REPORTS, ROOT


def select_runs(technique="exafs", alternatives=("realism_exafs_r1g10",)):
    """Choose between alternative realism runs of one technique by their best validation checkpoint score (the score
    used for checkpoint selection during training; validation split only, never the held-out split). The winner is
    moved to artifacts/models/realism_<t>/ and the other runs are kept next to it with a _run<i> suffix."""
    import os
    from .paths import load_yaml

    def summary(name):
        # the run's saved generator (its best checkpoint under the training rule) and its validation diagnostics
        h = json.loads((MODELS / name / "history.json").read_text())
        ck = torch.load(MODELS / name / "generator_ema.pt", map_location="cpu", weights_only=False)
        x, step = ck["diag"], ck["step"]
        s = abs(x["c2st_acc"] - 0.5) + 0.05 * x.get("feat_fd_ratio", 0.0)
        cfg = load_yaml(MODELS / name / "config.yaml")
        return dict(run=name, r1_gamma=cfg["r1_gamma"], best_score=s, best_step=step, last_step=h[-1]["step"],
                    best_val_c2st=x["c2st_acc"], best_val_feat_fd_ratio=x.get("feat_fd_ratio"))
    base = f"realism_{technique}"
    runs = [summary(base)] + [summary(a) for a in alternatives if (MODELS / a / "generator_ema.pt").exists()]
    win = min(runs, key=lambda r: r["best_score"])
    if win["run"] != base:
        for i, r in enumerate(runs, 1):
            if r["run"] != win["run"]:
                os.replace(MODELS / r["run"], MODELS / f"{base}_run{i}")
                r["kept_as"] = f"{base}_run{i}"
        os.replace(MODELS / win["run"], MODELS / base)
        win["kept_as"] = base
    txt = "; ".join(f"run with R1 gamma {r['r1_gamma']:g}: best validation score {r['best_score']:.3f} at step "
                    f"{r['best_step']} (C2ST {r['best_val_c2st']:.3f}), stopped at {r['last_step']}" for r in runs)
    rec = dict(runs=runs, selected=win["run"], summary=f"{len(runs)} training runs ({txt}); selected the run with R1 "
               f"gamma {win['r1_gamma']:g} on the validation score")
    (REPORTS / f"{base}_selection.json").write_text(json.dumps(rec, indent=1, default=float))
    print(rec["summary"], flush=True)
    return rec


ALTERNATIVE_RUNS = {"exafs": ("realism_exafs_r1g10",), "pdf": ("realism_pdf_r1g10",)}


def select_all():
    """Runs the selection once per technique; an existing reports/realism_<t>_selection.json means it is done (the
    alternative run has already been moved, so selecting again would lose the record of the other run)."""
    out = {}
    for t, alts in ALTERNATIVE_RUNS.items():
        rec = REPORTS / f"realism_{t}_selection.json"
        if rec.exists():
            print(f"{t}: selection already made ({rec.name})", flush=True)
            out[t] = json.loads(rec.read_text())
        else:
            out[t] = select_runs(t, alts)
    return out


def tidy():
    """Keep only the shipped files in artifacts/models/realism_<t>/ (generator_ema.pt + hash, config, history, model
    card); intermediate EMA checkpoints, resume states and the runs not selected move to artifacts/models/work/."""
    keep = {"generator_ema.pt", "generator_ema.sha256", "config.yaml", "history.json", "MODEL_CARD.md"}
    work = MODELS / "work"
    for t in ("xrd", "pdf", "exafs"):
        d = MODELS / f"realism_{t}"
        dst = work / f"realism_{t}_checkpoints"
        dst.mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():
            if f.name not in keep:
                shutil.move(str(f), str(dst / f.name))
        rec = REPORTS / f"realism_{t}_selection.json"
        if rec.exists():
            sel = json.loads(rec.read_text())
            for r in sel["runs"]:
                k = r.get("kept_as")
                if k and k != f"realism_{t}" and (MODELS / k).exists():
                    shutil.move(str(MODELS / k), str(work / k))
                    r["kept_as"] = f"work/{k}"
            rec.write_text(json.dumps(sel, indent=1, default=float))
    print(sorted(p.name for p in MODELS.iterdir()), flush=True)


def diagnostics(*techniques, predictor_ckpt=None, device="cuda:0"):
    """Per-technique results are cached in reports/gan_diag_<t>.json together with the SHA-256 of the generator and
    of the encoder checkpoint, and reused only if both are unchanged. With technique arguments, only those are
    (re)computed and the report is written once all three are available."""
    from .diagnostics import run
    from .paths import sha256_file
    from .reports import gan_diagnostics_report
    if predictor_ckpt is None:
        # Frechet distances use the encoder of predictor (a), seed 1: it never saw GAN output, so the realism
        # diagnostics do not depend on the downstream decision
        predictor_ckpt = MODELS / "work" / "predictor_a_s1" / "best.pt"
    esha = sha256_file(predictor_ckpt)
    res = []
    for t in ("xrd", "pdf", "exafs"):
        cache = REPORTS / f"gan_diag_{t}.json"
        gsha = sha256_file(MODELS / f"realism_{t}" / "generator_ema.pt")
        if t not in techniques and cache.exists():
            r = json.loads(cache.read_text())
            if r.get("generator_sha256") == gsha and r.get("encoder_sha256") == esha:
                res.append({k: v for k, v in r.items() if k != "rows"})
                print(f"{t}: reusing {cache.name} (generator and encoder unchanged)", flush=True)
                continue
        if techniques and t not in techniques:
            print(f"{t}: not computed in this call", flush=True)
            continue
        r = run(t, str(predictor_ckpt), device=device)
        r.update(generator_sha256=gsha, encoder_sha256=esha)
        rows = json.loads(cache.read_text())["rows"]
        cache.write_text(json.dumps(dict(r, rows=rows), indent=1, default=float))
        res.append(r)
        print(json.dumps(r, indent=1, default=float), flush=True)
    if len(res) == 3:
        gan_diagnostics_report(res)
        (REPORTS / "gan_diagnostics.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def downstream():
    from .downstream import main
    decision, best_seed = main()
    win = decision["ship"]
    src = MODELS / "work" / f"predictor_{win}_s{best_seed}"
    dst = MODELS / "predictor"
    dst.mkdir(parents=True, exist_ok=True)
    ck = torch.load(src / "best.pt", map_location="cpu", weights_only=False)
    torch.save(dict(model=ck["model"], stats=ck["stats"], variant=win, seed=best_seed, step=ck["step"],
                    val=ck.get("val")), dst / "model.pt")
    shutil.copyfile(src / "config.json", dst / "config.json")
    (dst / "SHIPPED.json").write_text(json.dumps(dict(variant=win, seed=best_seed, decision=decision), indent=1,
                                                 default=str))
    print("shipped", win, "seed", best_seed, decision, flush=True)
    return decision, best_seed


def cards():
    from .cards import dataset_card, predictor_model_card, realism_model_card
    dataset_card(ROOT / "artifacts" / "data", "full build, 50,000 samples")
    shipped = json.loads((MODELS / "predictor" / "SHIPPED.json").read_text())
    ds = json.loads((REPORTS / "downstream_comparison.json").read_text())
    diag = {d["technique"]: d for d in json.loads((REPORTS / "gan_diagnostics.json").read_text())}
    for t in ("xrd", "pdf", "exafs"):
        z = np.load(ROOT / "artifacts" / "data" / "real" / f"{t}.npz", allow_pickle=False)
        ok = z["twin_ok"] if "twin_ok" in z.files else np.ones(len(z["x"]), bool)
        sp = z["split"]
        counts = (f"{int(((sp == 'train') & ok).sum())} train, {int(((sp == 'val') & ok).sum())} validation, "
                  f"{int((sp == 'eval').sum())} held out ({int(((sp == 'eval') & ok).sum())} with usable twins); sources: "
                  + ", ".join(sorted(set(str(s) for s in z["source"]))))
        hist = MODELS / f"realism_{t}" / "history.json"
        h = json.loads(hist.read_text()) if hist.exists() else []
        gen = torch.load(MODELS / f"realism_{t}" / "generator_ema.pt", map_location="cpu", weights_only=False) \
            if (MODELS / f"realism_{t}" / "generator_ema.pt").exists() else {}
        from .paths import load_yaml
        rc = load_yaml(MODELS / f"realism_{t}" / "config.yaml")
        info = (f"selected checkpoint step {gen.get('step', 'n/a')}, last evaluated step {h[-1]['step'] if h else 'n/a'}; "
                f"batch {rc['batch']}, Adam {rc['lr_g']:g} / {rc['lr_d']:g}, betas (0, 0.99), EMA {rc['ema']}, "
                f"R1 gamma {rc['r1_gamma']:g} (every {rc['r1_every']} steps), preservation weight {rc['w_pres']:g}, "
                f"magnitude weight {rc['w_mag']:g}, bf16")
        sel = REPORTS / f"realism_{t}_selection.json"
        if sel.exists():
            info += "; " + json.loads(sel.read_text())["summary"]
        realism_model_card(t, diag.get(t, {}), shipped["variant"] == "b", counts, info)
    cfg = json.loads((MODELS / "predictor" / "config.json").read_text())["cfg"]
    ck = torch.load(MODELS / "predictor" / "model.pt", map_location="cpu", weights_only=False)
    splits = json.loads((ROOT / "artifacts" / "data" / "splits.json").read_text())
    counts = (f"{len(splits['train'])} clean + {len(splits['fault']['train'])} fault training samples, "
              f"{len(splits['val'])} + {len(splits['fault']['val'])} validation")
    predictor_model_card(shipped["variant"], shipped["seed"], cfg, ck.get("val") or {}, ds["rows"], ds["decision"], counts)
    print("cards written", flush=True)


if __name__ == "__main__":
    {"select": select_all, "diagnostics": diagnostics, "downstream": downstream, "cards": cards, "tidy": tidy}[sys.argv[1]](*sys.argv[2:])
