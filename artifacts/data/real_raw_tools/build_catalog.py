"""Build D:\\GAN\\artifacts\\data\\real_raw\\catalog.csv from all downloaded sources (parses data only).

Usage: python -I build_catalog.py
Outputs: real_raw/catalog.csv, real_raw/catalog_excluded_duplicates.csv, real_raw_tools/catalog_summary.json
"""
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, r"D:/GAN/artifacts/data/real_raw_tools")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from catalog_common import COLUMNS, ROOT  # noqa: E402
import build_catalog_xas as X  # noqa: E402
import build_catalog_xas2 as X2  # noqa: E402
import build_catalog_xrd_pdf as XP  # noqa: E402

LARCH_BEAMLINE_EDGES = {  # from file names / headers of examples/xafsdata/beamlines (checked by hand)
    "APS10BM_2019.dat": ("Mn K", "Mn_MnO2_5B_kapton (header)"), "APS12BM_2019.dat": ("Mn K", "#C exafs_edge_energy 6.539"),
    "APS13ID_2008.dat": ("Zn K", "energy range"), "APS13ID_2019.dat": ("Mn K", "energy range 6389-6537 eV (pre-edge/XANES only)"),
    "APS20BM_2001.dat": ("Fe K", "identical to feo_exafs_pnccat2001.dat"), "APS20ID_2018.dat": ("Au L3", "energy range (inferred)"),
    "APS20ID_2022.dat": ("Pb L3", "energy range (inferred)"), "APS9BM_2006.dat": ("S K", "Glut_red_powder (glutathione)"),
    "APS9BM_2019.dat": ("Mn K", "energy range"), "CLSHXMA.dat": ("As K", "energy range"),
    "ESRF_BM08_LISA_2021.dat": ("Ge K", "energy range"), "ESRF_SNBL_2013.dat": ("Ge K", "GeO2 (header); energy column not first"),
    "NSLS6BM_2019.dat": ("V K", "energy range"), "NSLS8ID_2019.dat": ("As K", "energy range"),
    "NSLS_XDAC_2011.dat": ("Fe K", "Fe_trilayer (header)"), "PF9A_2022.dat": ("Fe K", "Fe010 (header); 9809 angle-based format"),
    "PFBL12C_2005.dat": ("Hg L3", "Hg:H2Cys (header); 9809 angle-based format"), "SLS_PHOENIX_2023.dat": ("Al K", "MgKa/AlKa counters, tender range (inferred)"),
    "SSRL1_2006.dat": ("unknown", "SSRL EXAFS Data Collector raw file"), "SSRLmicro_2008.dat": ("Ge K", "energy range (inferred)"),
}
FORMULA_RE = re.compile(r"^(\(?[A-Z][a-z]?\d*(\.\d+)?\)?\d*)+(\s*[.·]\s*\d*H2O)?$")


def file_md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main():
    rows, excluded = [], []
    r_xaslib, f1 = X.src_xaslib()
    r_bmm, f2 = X.src_bmm()
    r_lisa, f3 = X.src_lisa()
    known = {**f1, **f2, **f3}
    r_larch, d_larch = X.src_larch(known)
    r_cls, d_cls = X.src_cls(known)
    excluded += [dict(item_id=a, duplicate_of=b, reason="same scan already catalogued (energy-array fingerprint)") for a, b in d_larch]
    excluded += [dict(item_id=a, duplicate_of=b, reason="CLS XASDB re-distribution of a spectrum from another catalogued source")
                 for a, b in d_cls]
    # BMM: formula-like sample names -> composition
    for r in r_bmm:
        if not r["composition"] and FORMULA_RE.match(r["material"].replace(" ", "")):
            r["composition"] = r["material"].replace(" ", "")
    # Larch fixes: beamline raw-file edges, sample types, de-duplicate identical files
    seen_md5 = {}
    larch_keep = []
    for r in r_larch:
        p = ROOT / r["file_path"]
        h = file_md5(p)
        if h in seen_md5:
            excluded.append(dict(item_id=r["item_id"], duplicate_of=seen_md5[h], reason="identical file content"))
            continue
        seen_md5[h] = r["item_id"]
        if p.name in LARCH_BEAMLINE_EDGES and "/beamlines/" in r["file_path"]:
            ed, why = LARCH_BEAMLINE_EDGES[p.name]
            r["edge_or_wavelength"] = ed
            r["notes"] = re.sub(r"kmax~[\d.]+ A-1; ", "", r["notes"]) + f"; edge from {why}"
            r["material"] = f"raw beamline example file {p.stem}"
        if p.name == "fe_xanes_8ch.xdi":
            r["edge_or_wavelength"] = "Fe K"
            r["notes"] += "; Fe K pre-edge/XANES region (8 fluorescence channels)"
        if r["composition"] == "Cu" or p.name.startswith(("cu_", "cu.")) and "glass" not in p.name:
            r["sample_type"] = "foil"
            r["composition"] = "Cu"
            r["material"] = f"Cu foil ({p.name})"
        larch_keep.append(r)
    rows += r_xaslib + r_bmm + r_lisa + larch_keep + r_cls
    rows += X.src_mdr()
    rows += X2.src_htpem_17131699() + X2.src_ptfeox_8041702() + X2.src_ucl() + X2.src_cardiff_xas()
    wr = X2.src_wright()
    exafs_ids = {r["item_id"].split("Wright2026_EXAFS_data_")[1]: r["item_id"] for r in wr if "Wright2026_EXAFS_data_" in r["item_id"]}
    for r in list(wr):
        if "Wright2026_XANES_data_" in r["item_id"]:
            lab = r["item_id"].split("Wright2026_XANES_data_")[1]
            excluded.append(dict(item_id=r["item_id"], duplicate_of=exafs_ids.get(lab, "Wright2026_EXAFS_data (same sample)"),
                                 reason="same measurement as the EXAFS-file column (both normalised mu(E) on the same grid)"))
            wr.remove(r)
    for r in wr:
        if r["source"] == "mendeley_9bhtr2w54w":
            r["notes"] = re.sub(r"; from EXAFS file \(.*\)$", "; normalised mu(E) from Wright2026_EXAFS_data.csv (the XANES csv holds "
                                "the same scans; excluded as duplicates)", r["notes"])
    rows += wr + X2.src_pt_fluids() + X2.src_alpha_pto2() + X2.src_ni3ga() + X2.src_coo_herfd()
    rows += XP.src_htpem_xrd() + XP.src_small_xrd() + XP.src_pdf()
    rows += XP.src_opxrd()
    # unique ids + sanity
    cnt = Counter()
    for r in rows:
        cnt[r["item_id"]] += 1
        if cnt[r["item_id"]] > 1:
            r["item_id"] = f"{r['item_id']}__{cnt[r['item_id']]}"
        for k in COLUMNS:
            v = r.get(k, "")
            r[k] = "" if v is None else str(v).replace("\r", " ").replace("\n", " ").strip()
        assert (ROOT / r["file_path"]).exists(), r["file_path"]
    with open(ROOT / "catalog.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(ROOT / "catalog_excluded_duplicates.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["item_id", "duplicate_of", "reason"])
        w.writeheader()
        w.writerows(excluded)
    # ---------------- summary
    pt_re = re.compile(r"(?<![A-Za-z])Pt(?![a-z])")
    s = {"n_rows": len(rows), "by_technique": Counter(r["technique"] for r in rows), "by_source": defaultdict(Counter)}
    for r in rows:
        s["by_source"][r["source"]][r["technique"]] += 1
    xas = [r for r in rows if r["technique"] == "xas"]
    s["xas_by_edge"] = Counter(r["edge_or_wavelength"] for r in xas)
    s["xas_target_edges"] = {e: sum(1 for r in xas if r["edge_or_wavelength"] == e) for e in ("Pt L3", "Co K", "Ni K", "Fe K", "Cu K")}
    s["xas_target_edges_by_source"] = {e: Counter(r["source"] for r in xas if r["edge_or_wavelength"] == e) for e in ("Pt L3", "Co K", "Ni K", "Fe K", "Cu K")}
    s["xas_target_edges_by_sample_type"] = {e: Counter(r["sample_type"] for r in xas if r["edge_or_wavelength"] == e) for e in ("Pt L3", "Co K", "Ni K", "Fe K", "Cu K")}
    ts = [r for r in xas if "TIME-RESOLVED MATRIX" in r["notes"]]
    s["xas_time_series_files"] = [(r["item_id"], r["edge_or_wavelength"], re.search(r"(\d+) spectra|(\d+) column pairs", r["notes"]).group(0)) for r in ts]
    s["xas_x_kind"] = Counter(r["x_kind"] for r in xas)
    for tech in ("xrd", "pdf"):
        tr = [r for r in rows if r["technique"] == tech]
        s[f"{tech}_pt_related"] = sum(1 for r in tr if pt_re.search(r["composition"] + " " + r["material"]))
        s[f"{tech}_pt_related_by_source"] = Counter(r["source"] for r in tr if pt_re.search(r["composition"] + " " + r["material"]))
        s[f"{tech}_other"] = len(tr) - s[f"{tech}_pt_related"]
        s[f"{tech}_wavelengths"] = Counter(r["edge_or_wavelength"] for r in tr).most_common(25)
        s[f"{tech}_x_kind"] = Counter(r["x_kind"] for r in tr)
    s["xrd_cu_ka"] = sum(1 for r in rows if r["technique"] == "xrd" and r["edge_or_wavelength"].startswith("CuKa"))
    s["opxrd_cu_ka"] = sum(1 for r in rows if r["source"] == "opxrd" and r["edge_or_wavelength"].startswith("CuKa"))
    s["pdf_by_x_kind_pt"] = Counter(r["x_kind"] for r in rows if r["technique"] == "pdf" and pt_re.search(r["composition"] + " " + r["material"]))
    s["licences"] = Counter(r["licence"] for r in rows).most_common()
    s["n_excluded_duplicates"] = len(excluded)
    json.dump(s, open(ROOT.parent / "real_raw_tools/catalog_summary.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=dict)
    print(json.dumps(s, indent=1, ensure_ascii=False, default=dict)[:12000])


if __name__ == "__main__":
    main()
