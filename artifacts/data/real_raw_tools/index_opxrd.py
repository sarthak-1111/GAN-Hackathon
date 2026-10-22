"""Index all opXRD pattern JSON files (data only; parsed with json, nothing executed).

Usage: python -I index_opxrd.py <extracted_dir> <out_csv>
Columns: rel_path, group, institution, contributor, original_format, measurement_date,
         wavelength1, wavelength2, is_simulated, n_points, tt_min, tt_max, step_med,
         n_phases, elements, phase_formulas, spacegroups, temp_K, crystallite_size_nm, tags
"""
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def loads_maybe(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except ValueError:
            return x
    return x


def phase_info(ph):
    ph = loads_maybe(ph)
    if not isinstance(ph, dict):
        return set(), "", ""
    basis = loads_maybe(ph.get("basis")) or []
    els = Counter()
    for a in basis:
        a = loads_maybe(a)
        if isinstance(a, dict):
            sym = re.sub(r"[^A-Za-z]", "", str(a.get("symbol", "")))
            if sym:
                occ = a.get("occupancy") or 1
                try:
                    els[sym] += float(occ)
                except (TypeError, ValueError):
                    els[sym] += 1
    comp = ph.get("chemical_composition")
    formula = str(comp) if comp else "".join(f"{e}{round(n, 2) if abs(n - round(n)) > 1e-6 else int(round(n))}"
                                            for e, n in sorted(els.items()))
    sg = ph.get("spacegroup")
    return set(els), formula, "" if sg is None else str(sg)


def main():
    root = Path(sys.argv[1])
    out = sys.argv[2]
    files = sorted(root.rglob("*.json"))
    print("files", len(files), flush=True)
    cols = ["rel_path", "group", "institution", "contributor", "original_format", "measurement_date",
            "wavelength1", "wavelength2", "is_simulated", "n_points", "tt_min", "tt_max", "step_med",
            "n_phases", "elements", "phase_formulas", "spacegroups", "temp_K", "crystallite_size_nm", "tags"]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i, f in enumerate(files):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except Exception as e:  # noqa: BLE001
                print("bad", f, e)
                continue
            tt = d.get("two_theta_values") or []
            md = loads_maybe(d.get("metadata")) or {}
            lb = loads_maybe(d.get("label")) or {}
            xi = loads_maybe(lb.get("xray_info")) or {}
            phases = lb.get("phases") or []
            els, forms, sgs = set(), [], []
            for ph in phases:
                e, fo, sg = phase_info(ph)
                els |= e
                forms.append(fo)
                sgs.append(sg)
            step = ""
            if len(tt) > 2:
                diffs = sorted(tt[j + 1] - tt[j] for j in range(0, len(tt) - 1, max(1, len(tt) // 200)))
                step = f"{diffs[len(diffs) // 2]:.5g}"
            rel = f.relative_to(root).as_posix()
            w.writerow([rel, rel.split("/")[0] + ("/" + rel.split("/")[1] if rel.count("/") > 1 else ""),
                        md.get("institution"), md.get("contributor_name"), md.get("original_file_format"),
                        md.get("measurement_date"), xi.get("primary_wavelength"), xi.get("secondary_wavelength"),
                        lb.get("is_simulated"), len(tt), f"{min(tt):.4f}" if tt else "", f"{max(tt):.4f}" if tt else "",
                        step, len(phases), " ".join(sorted(els)), "; ".join(forms), "; ".join(sgs),
                        lb.get("temp_K"), lb.get("crystallite_size_nm"), "|".join(map(str, md.get("tags") or []))])
            if i % 5000 == 0:
                print(i, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
