"""XAS sources -> catalog rows. Imported by build_catalog.py."""
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"D:/GAN/artifacts/data/real_raw_tools")
from catalog_common import (ROOT, edge_energy, edge_from_data, edge_label, guess_sample_type, infer_edge_from_range,  # noqa: E402
                            mode_from_names, numeric_block, parse_temp, parse_xdi, read_text, rel,
                            write_two_col, xas_notes)

CC0 = "CC0-1.0"
CCBY = "CC-BY-4.0"


def efp(e):
    """energy-array fingerprint used to detect re-distributed duplicates of the same scan."""
    e = np.asarray(e, float)
    if e.size < 5:
        return None
    return (len(e), round(float(e[0]), 1), round(float(e[len(e) // 2]), 1), round(float(e[-1]), 1))


def row(**kw):
    kw.setdefault("temperature_K", "")
    kw.setdefault("notes", "")
    return kw


# ----------------------------------------------------------------------------------------------------------
def xdi_row(path, source, url, licence, citation, item_id, extra_notes="", material_override=None):
    meta, names, data = parse_xdi(path)
    el = meta.get("Element.symbol", "").strip()
    edge = meta.get("Element.edge", "").strip()
    e = data[:, 0] if data.size else np.array([])
    unit_note = ""
    c1 = (names[0] if names else "").lower()
    if "kev" in c1:
        e = e * 1000.0
        unit_note = "energy column in keV"
    if not el and e.size:
        el, edge, _ = infer_edge_from_range(float(e.min()), float(e.max()))
        unit_note += "; edge inferred from energy range" if el else ""
    name = meta.get("Sample.name", "") or Path(path).stem
    formula = meta.get("Sample.formula", "")
    tK, tnote = parse_temp(meta.get("Sample.temperature", ""))
    mode = mode_from_names(names, meta.get("Scan.plot_hint", "") + " " + meta.get("Detector.I1", ""))
    bl = " ".join(x for x in (meta.get("Facility.name", ""), meta.get("Beamline.name", "")) if x)
    notes = xas_notes(mode, e, el, edge, "; ".join(x for x in (f"beamline={bl}" if bl else "", "columns=" + ",".join(names[:8]),
                                                                    tnote, unit_note.strip("; "), extra_notes) if x))
    return row(item_id=item_id, source=source, source_record_url=url, file_path=rel(path), licence=licence,
               technique="xas", edge_or_wavelength=edge_label(el, edge), x_kind="energy_eV",
               material=material_override or name, composition=formula,
               sample_type=guess_sample_type(name + " " + Path(path).stem, formula, meta.get("Sample.prep", "")),
               temperature_K=tK, notes=notes, citation=citation), efp(e)


def src_xaslib():
    rows, fps = [], {}
    idx = json.load(open(ROOT / "xaslib/_index.json", encoding="utf-8"))
    for rec in idx:
        p = ROOT / "xaslib" / rec["file"]
        r, fp = xdi_row(p, "xaslib", f"https://xaslib.xrayabsorption.org/spectrum/{rec['id']}/", CC0,
                        f"IXAS X-ray Absorption Data Library (XASLIB) spectrum {rec['id']} '{rec['name']}', "
                        f"https://xaslib.xrayabsorption.org (International X-ray Absorption Society); CC0",
                        f"xaslib_{rec['id']:04d}", extra_notes=f"xaslib beamline label={rec['beamline']}")
        if not r["edge_or_wavelength"].strip():
            r["edge_or_wavelength"] = edge_label(rec["element"], rec["edge"])
        rows.append(r)
        if fp:
            fps[fp] = r["item_id"]
    return rows, fps


def src_bmm():
    rows, fps = [], {}
    base = ROOT / "bmm_standards/extracted/bmm-standards-main/Data"
    for p in sorted(base.rglob("*.xdi")):
        r, fp = xdi_row(p, "bmm_standards_nist", "https://doi.org/10.18434/mds2-4032",
                        "NIST public-domain data (17 U.S.C. 105) with worldwide royalty-free licence, see https://www.nist.gov/open/license",
                        "Bruce Ravel (2025), A collection of X-ray Absorption Spectroscopy data of metals, stable oxides, and "
                        "other common compounds measured at NIST's Beamline for Materials Measurement, Version 1.0.0, NIST, "
                        "https://doi.org/10.18434/mds2-4032",
                        "bmm_" + re.sub(r"[^A-Za-z0-9]+", "_", p.stem))
        if "ln(It/Ir)" in read_text(p)[:4000] and "foil" in r["material"].lower():
            r["notes"] += "; xmu here is the reference-foil channel ln(It/Ir)"
        rows.append(r)
        if fp:
            fps[fp] = r["item_id"]
    return rows, fps


def src_lisa():
    rows, fps = [], {}
    live = ROOT / "lisa_xasdb/live_site"
    names_live = {p.name for p in live.rglob("*.xdi")}
    for p in sorted(live.rglob("*.xdi")):
        el = p.parent.name
        r, fp = xdi_row(p, "lisa_xasdb", f"https://lisa.iom.cnr.it/xasdb/{el}.html",
                        "CC-BY-4.0 (Zenodo record 10.5281/zenodo.10778068 of the LISA XAS Database; site footer: 'Copyright Public Domain')",
                        "A. Puri, LISA XAS Database, BM08-LISA (CNR-IOM CRG @ ESRF), https://lisa.iom.cnr.it/xasdb/ , "
                        "doi:10.5281/zenodo.10778068", "lisa_" + re.sub(r"[^A-Za-z0-9]+", "_", f"{el}_{p.stem}"))
        rows.append(r)
        if fp:
            fps[fp] = r["item_id"]
    # zenodo v1.0.2 snapshot: only files not on the live site
    for p in sorted((ROOT / "lisa_xasdb/zenodo_v1.0.2_extracted").rglob("*.xdi")):
        if p.name in names_live:
            continue
        r, fp = xdi_row(p, "lisa_xasdb", "https://doi.org/10.5281/zenodo.10778098", CCBY,
                        "A. Puri, LISA XAS Database v1.0.2, Zenodo, doi:10.5281/zenodo.10778098",
                        "lisa_zen_" + re.sub(r"[^A-Za-z0-9]+", "_", p.stem))
        if fp in fps:
            continue
        rows.append(r)
        if fp:
            fps[fp] = r["item_id"]
    return rows, fps


def src_larch(known_fps):
    rows, dups = [], []
    base = ROOT / "github_xraylarch_examples"
    lic = "MIT (xraylarch repository licence covering example data)"
    cit = "Larch example data, github.com/xraypy/xraylarch (examples/xafsdata), commit " + read_text(base / "COMMIT.txt").split()[0][:10]
    skip = {"nonuniform.chi", "generic_columns_no_header.dat", "cu_chie.dat"}
    for p in sorted(base.rglob("*")):
        if not p.is_file() or p.name in ("LICENSE", "COMMIT.txt", "filelist.txt") or p.name in skip:
            continue
        url = "https://github.com/xraypy/xraylarch/tree/master/examples/" + rel(p).split("github_xraylarch_examples/")[1]
        iid = "larch_" + re.sub(r"[^A-Za-z0-9]+", "_", rel(p).split("github_xraylarch_examples/")[1])
        text = read_text(p)
        if p.suffix.lower() == ".xdi" or text.startswith("# XDI") or text.startswith("#XDI"):
            r, fp = xdi_row(p, "github_xraylarch_examples", url, lic, cit, iid)
            if fp in known_fps:
                dups.append((iid, known_fps[fp]))
                continue
            rows.append(r)
            continue
        is_chi = p.suffix.lower() == ".chi" or "_chi" in p.name
        arr = numeric_block(text)
        if arr.size == 0 or arr.shape[1] < 2:
            continue
        x = arr[:, 0]
        if is_chi:
            el, edge = ("Cu", "K") if p.name.startswith("cu") else ("", "")
            rows.append(row(item_id=iid, source="github_xraylarch_examples", source_record_url=url, file_path=rel(p),
                            licence=lic, technique="xas", edge_or_wavelength=edge_label(el, edge), x_kind="k",
                            material=p.stem, composition="Cu" if el == "Cu" else "", sample_type="foil" if el == "Cu" else "other",
                            temperature_K="10" if "10k" in p.name else "",
                            notes=f"chi(k) file; npts={len(x)}; k={x.min():.2f}-{x.max():.2f} A-1; columns: k chi ...",
                            citation=cit))
            continue
        # energy-like first column?  (SPEC/beamline raw files may have energy in another column)
        ecol = None
        for j in range(arr.shape[1]):
            c = arr[:, j]
            if 1000 < np.nanmedian(c) < 40000 and np.all(np.diff(c[np.isfinite(c)]) > -5):
                ecol = j
                break
        if ecol is None:
            # keV?
            for j in range(arr.shape[1]):
                c = arr[:, j]
                if 1.0 < np.nanmedian(c) < 40 and np.all(np.diff(c) > -0.005) and (c.max() - c.min()) > 0.05:
                    arr[:, j] = c * 1000
                    ecol = j
                    break
        if ecol is None:
            rows.append(row(item_id=iid, source="github_xraylarch_examples", source_record_url=url, file_path=rel(p),
                            licence=lic, technique="xas", edge_or_wavelength="unknown", x_kind="energy_eV",
                            material=p.stem, composition="", sample_type="other",
                            notes="raw beamline file (angle/encoder based or unusual layout); needs beamline-specific parsing, e.g. larch.io.read_ascii",
                            citation=cit))
            continue
        e = arr[:, ecol]
        name_el = re.match(r"^([A-Z][a-z]?)[-_]", p.name)
        hint = {"cu": "Cu", "fe": "Fe", "ni": "Ni", "pt": "Pt", "mn": "Mn", "sno2": "Sn", "znse": "Zn", "scorodite": "As",
                "se": "Se", "v": "V", "feo": "Fe", "fexanes": "Fe"}
        cand = None
        for k, v in hint.items():
            if p.name.lower().startswith(k):
                cand = [v]
        el, edge, _ = infer_edge_from_range(float(np.nanmin(e)), float(np.nanmax(e)), cand)
        if cand is None or el is None:
            el2, edge2, ej = edge_from_data(e, arr, ecol)
            if el2:
                el, edge = el2, edge2
            elif el is None:
                el, edge, _ = infer_edge_from_range(float(np.nanmin(e)), float(np.nanmax(e)))
        mode = mode_from_names([], text[:3000])
        tmatch = re.search(r"cu_(\d+)k", p.name.lower())
        rows.append(row(item_id=iid, source="github_xraylarch_examples", source_record_url=url, file_path=rel(p), licence=lic,
                        technique="xas", edge_or_wavelength=edge_label(el or "", edge or ""), x_kind="energy_eV",
                        material=p.stem, composition="Cu" if p.name.startswith("cu") and "glass" not in p.name else "",
                        sample_type=guess_sample_type(p.stem),
                        temperature_K=tmatch.group(1) if tmatch else "",
                        notes=xas_notes(mode, e, el, edge, f"energy in column {ecol+1}; edge inferred from energy range/file name; "
                                        f"{'raw beamline-format file (counters, not mu); ' if 'beamlines' in str(p) else ''}"
                                        f"{arr.shape[1]} numeric columns"),
                        citation=cit))
    return rows, dups


# ----------------------------------------------------------------------------------------------------------
def src_cls(known_fps):
    """CLS XASDB JSON -> derived text files; skip spectra re-distributed from XASLIB/BMM/LISA."""
    rows, dups = [], []
    XASLIB_DOI, BMM_DOI = "10.71622/XASLIB_535X-8J62", "10.18434/mds2-4032"
    for f in sorted(glob.glob(str(ROOT / "cls_xasdb/json/*/*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        h = d.get("header", {}) or {}
        sid = Path(f).stem
        doi = (h.get("Publication.DOI") or "").strip()
        e = np.asarray(d.get("energy") or [], float)
        mu = np.asarray(d.get("Mu") or [], float)
        iid = "cls_" + sid
        if XASLIB_DOI in doi:
            dups.append((iid, "xaslib (by DOI)"))
            continue
        if BMM_DOI in doi:
            dups.append((iid, "bmm_standards_nist (by DOI)"))
            continue
        fp = efp(e) if e.size else None
        if fp in known_fps:
            dups.append((iid, known_fps[fp]))
            continue
        el = h.get("Element.symbol", sid.split("_")[0])
        edd = h.get("Element.edge") or {}
        edge = list(edd.keys())[0] if isinstance(edd, dict) and edd else str(edd)
        if e.size == 0 or mu.size != e.size:
            continue
        cols, ys = ["energy_eV", "mu"], [mu]
        for k in ("I0", "Itrans", "Ifluor", "Irefer", "TEY"):
            v = d.get(k) or []
            if len(v) == len(e):
                cols.append(k)
                ys.append(np.asarray(v, float))
        dest = ROOT / "cls_xasdb/derived" / el / f"{sid}.dat"
        hdr = [f"derived from CLS XASDB JSON {rel(f)} (POST /xasgetcompounds id={sid})"] + \
              [f"{k}: {v}" for k, v in h.items() if not isinstance(v, (dict, list))] + [f"Element.edge: {edge}"]
        write_two_col(dest, e, ys, hdr, cols)
        name = h.get("Sample.name", d.get("title", sid))
        formula = h.get("Sample.formula", "")
        tK, tnote = parse_temp(h.get("Sample.temperature", ""))
        mode = (h.get("Beamline.scan_mode") or "").strip()
        bl = f"{h.get('Facility.name', '')} {h.get('Beamline.name', '')}".strip()
        authors = (h.get("Publication.authors") or "").strip()
        rows.append(row(item_id=iid, source="cls_xasdb", source_record_url=f"https://xasdb.lightsource.ca/search?id={sid}",
                        file_path=rel(dest), licence=(h.get("Scan.license") or "CC BY 4.0").replace("CC BY 4.0", CCBY),
                        technique="xas", edge_or_wavelength=edge_label(el, edge), x_kind="energy_eV", material=name,
                        composition=formula, sample_type=guess_sample_type(name, formula, h.get("Sample.prep", "")),
                        temperature_K=tK,
                        notes=xas_notes(mode.lower(), e, el, edge, f"beamline={bl}; prep={h.get('Sample.prep', '')}; "
                                        f"raw JSON={rel(f)}; mu as provided by XASDB (Mu field); {tnote}".strip("; ")),
                        citation=f"{authors} 'XAS Database Data', Canadian Light Source XASDB, sample ID {h.get('Sample.id', sid)}"
                                 f"{', ' + doi if doi else ''}; Spasyuk D. (2025) arXiv:2509.13566"))
    return rows, dups


# ----------------------------------------------------------------------------------------------------------
def src_mdr():
    rows = []
    idx = {x["doi"]: x for x in json.load(open(ROOT / "mdr_xafs_db/_datacite_index.json", encoding="utf-8"))}
    for d in sorted((ROOT / "mdr_xafs_db/extracted").iterdir()):
        if not d.is_dir():
            continue
        doi = "10.48505/" + d.name.replace("nims_", "nims.")
        rec = idx.get(doi, {})
        mfile = d / "metadata.all.json"
        if not mfile.exists():
            continue
        md = json.load(open(mfile, encoding="utf-8"))
        samp = (md.get("sample") or [{}])[0]
        meas = md.get("measurement", {})
        meth = meas.get("method", {})
        edge_s = meth.get("absorption_edge", "")
        m = re.match(r"([A-Z][a-z]?)\s+([KLM]\d*)", edge_s or "")
        el, edge = (m.group(1), m.group(2)) if m else ("", "")
        det = meth.get("detection", "")
        atm = (meas.get("atmosphere") or {}).get("sample") or {}
        t, tu = atm.get("start_temperature"), atm.get("start_temperature_unit")
        tK, tnote = parse_temp(f"{t}{'' if tu in (None, 'None') else ' ' + str(tu)}" if t not in (None, "None") else "")
        di = md.get("data_info", {})
        fac = f"{di.get('facility', '')} {di.get('class_name', '') if di.get('class_name') not in (None, 'None') else ''}".strip()
        name = samp.get("name", "") or rec.get("title", "").replace("XAFS spectrum of ", "")
        formula = samp.get("chemical_formula", "") if samp.get("chemical_formula") not in (None, "None") else ""
        add = samp.get("additional_metadata", "") if samp.get("additional_metadata") not in (None, "None") else ""
        creators = ", ".join(rec.get("creators", []))
        cit = f"{creators} ({rec.get('year', '')}). {rec.get('title', '')}. MDR XAFS DB, NIMS. https://doi.org/{doi}"
        lic = "CC-BY-4.0" if rec.get("rights") == ["cc-by-4.0"] else ";".join(rec.get("rights", []))
        txts = sorted(p for p in d.glob("*.txt"))
        for p in txts:
            arr = numeric_block(read_text(p))
            if arr.size == 0 or arr.shape[1] < 2:
                continue
            is_ref = p.stem.endswith("_ref")
            e = arr[:, 0]
            if np.nanmedian(e) < 100:   # keV
                e = e * 1000
            iid = f"mdr_{d.name}" + ("_ref" if is_ref else "") + ("" if len(txts) <= 2 else "_" + re.sub(r"[^A-Za-z0-9]+", "_", p.stem)[-20:])
            raw = [q.name for q in d.iterdir() if q.suffix.lower() in (".dat", ".ex3") and q.stem.replace("_ref", "") == p.stem.replace("_ref", "")]
            if is_ref:
                ref_name = re.sub(r"^[A-Z]_|_Si\d{3}_\d+_ref$", "", p.stem)
                rows.append(row(item_id=iid, source="mdr_xafs_db", source_record_url=f"https://doi.org/{doi}", file_path=rel(p),
                                licence=lic, technique="xas", edge_or_wavelength=edge_label(el, edge), x_kind="energy_eV",
                                material=f"reference channel of dataset ({ref_name})", composition="",
                                sample_type="foil" if "foil" in p.stem.lower() else "other", temperature_K=tK,
                                notes=xas_notes("transmission (reference channel)", e, el, edge,
                                                f"facility={fac}; reference spectrum recorded with sample '{name}'; columns: energy_eV, mu*t; raw files: {','.join(raw)}"),
                                citation=cit))
            else:
                rows.append(row(item_id=iid, source="mdr_xafs_db", source_record_url=f"https://doi.org/{doi}", file_path=rel(p),
                                licence=lic, technique="xas", edge_or_wavelength=edge_label(el, edge), x_kind="energy_eV",
                                material=name, composition=formula, sample_type=guess_sample_type(name + " " + add, formula, add),
                                temperature_K=tK,
                                notes=xas_notes(det.lower(), e, el, edge,
                                                f"facility={fac}; sample form={add}; columns: energy_eV, mu*t (Athena-style 2-col); "
                                                f"raw files: {','.join(raw)} (9809/Rex2000 formats); {tnote}".strip("; ")),
                                citation=cit))
    return rows
