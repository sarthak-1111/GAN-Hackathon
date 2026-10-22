"""XAS sources (Pt-related deposits, small datasets) -> catalog rows + derived per-spectrum files."""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"D:/GAN/artifacts/data/real_raw_tools")
from catalog_common import (ROOT, edge_energy, edge_label, guess_sample_type, infer_edge_from_range,  # noqa: E402
                            numeric_block, parse_temp, read_text, rel, write_two_col, xas_notes)
from build_catalog_xas import row  # noqa: E402
import xlsx_read  # noqa: E402

CCBY = "CC-BY-4.0"


def safe(s):
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", str(s)).strip("_")[:80]


def edge_from_e(e, prefer=None):
    """Assign edge for an energy array: prefer listed elements (K, L3)."""
    emin, emax = float(np.nanmin(e)), float(np.nanmax(e))
    cands = prefer or ["Pt", "Co", "Ni", "Fe", "Cu", "Ga", "Mn", "Zn", "Ir", "Au", "Pd"]
    best = None
    for el in cands:
        for ed in ("K", "L3"):
            en = edge_energy(el, ed)
            if emin - 5 <= en <= emax:
                d = en - emin
                if best is None or d < best[0]:
                    best = (d, el, ed)
    if best:
        return best[1], best[2]
    el, ed, _ = infer_edge_from_range(emin, emax)
    return el or "", ed or ""


# ----------------------------------------------------------------------------------------------------------
def src_htpem_17131699():
    """Zenodo 17131699: XAS CSVs 0569-0575 (semicolon, decimal comma; 6 samples + foil per file)."""
    rows = []
    base = ROOT / "zenodo_17131699_PtCo_PtNi_HTPEM"
    url = "https://doi.org/10.5281/zenodo.17131699"
    cit = ("Buriánek J.D., Bouzek K., Paušová Š. (2025) Dataset for 'Activity and Degradation of Pt-Co and Pt-Ni Alloy "
           "Catalysts for Application in High-temperature PEM Fuel Cells', Zenodo, doi:10.5281/zenodo.17131699; "
           "article doi:10.1039/d5ey00279f")
    catmap = {"HIS": ("Pt/C (commercial)", "Pt", "catalyst"), "KET": ("Pt-Co/Ketjen Black (intermetallic)", "PtCo", "catalyst"),
              "RGO": ("Pt-Co/reduced graphene oxide (intermetallic)", "PtCo", "catalyst"), "GO": ("Pt-Co/reduced graphene oxide", "PtCo", "catalyst"),
              "NI": ("Pt-Ni/Ketjen Black (intermetallic)", "PtNi", "catalyst")}
    for p in sorted((base / "extracted").glob("VZ2_002_021_VSCHT_D_05[67]*_v1.csv")):
        num = int(re.search(r"_D_(\d+)_", p.name).group(1))
        if not 569 <= num <= 575:
            continue
        text = read_text(p).lstrip("\ufeff")
        lines = text.splitlines()
        labels = lines[2].split(";")
        # column pairs (x, y) sit at even/odd positions; the 'X' marker is sometimes missing (file 0573)
        sample_labels = {i: labels[i + 1] for i in range(0, len(labels) - 1, 2)
                         if labels[i].strip() in ("X", "") and labels[i + 1].strip() not in ("X", "")}
        data = []
        for ln in lines[3:]:
            parts = ln.split(";")
            try:
                vals = [float(x.replace(",", ".")) if x.strip() else np.nan for x in parts]
            except ValueError:
                continue
            data.append(vals)
        ncol = max(len(r) for r in data)
        arr = np.array([r + [np.nan] * (ncol - len(r)) for r in data])
        for xc, lab in sample_labels.items():
            if not lab.strip() or "curve translate" in lab:
                continue
            if xc + 1 >= arr.shape[1]:
                continue
            e, mu = arr[:, xc], arr[:, xc + 1]
            ok = np.isfinite(e) & np.isfinite(mu)
            e, mu = e[ok], mu[ok]
            if e.size < 20:
                continue
            m = re.match(r"([A-Za-z]+?)(\d+)_", lab)
            is_foil = "foil" in lab.lower()
            el, edge = edge_from_e(e, ["Pt", "Co", "Ni"])   # element from the energy range (labels are sometimes wrong)
            if is_foil and not lab.lower().startswith(el.lower()):
                lab_note = f" (column labelled '{lab}' but energy range is {el} {edge})"
            else:
                lab_note = ""
            if is_foil:
                mat, comp, st, hours = f"{el} foil (energy reference measured with the series){lab_note}", el, "foil", None
            else:
                key = m.group(1).upper() if m else lab.split("_")[0].upper()
                mat, comp, st = catmap.get(key, (key, "", "catalyst"))
                hours = int(m.group(2)) if m else None
            dest = base / "derived_xas" / f"{p.stem}__{safe(lab)}.dat"
            write_two_col(dest, e, [mu], [f"derived from {rel(p)} column pair '{lab}' (semicolon CSV, decimal comma)",
                                         "units: energy eV, mu a.u. (fluorescence, as deposited)"], ["energy_eV", "mu"])
            cond = "pristine" if hours == 0 else (f"leached {hours} h in 98 wt% H3PO4 at 180 C" if hours else "")
            rows.append(row(item_id=f"z17131699_{num}_{safe(lab)}", source="zenodo_17131699_PtCo_PtNi_HTPEM", source_record_url=url,
                            file_path=rel(dest), licence=CCBY, technique="xas", edge_or_wavelength=edge_label(el, edge),
                            x_kind="energy_eV", material=mat + (f"; {cond}" if cond else ""), composition=comp, sample_type=st,
                            temperature_K="", notes=xas_notes("fluorescence (PIPS), ex situ, DESY P64", e, el, edge,
                                                              f"orig file {p.name} col '{lab}'; label 'Pt_L2,3' in file = Pt L3 edge region"),
                            citation=cit))
    return rows


def src_ptfeox_8041702():
    rows = []
    base = ROOT / "zenodo_8041702_PtFeOx_operando_XAS/extracted/article2"
    url = "https://doi.org/10.5281/zenodo.8041702"
    cit = ("Open data for 'Platinum-Iron(II) Oxide Sites Directly Responsible for Preferential Carbon Monoxide Oxidation at "
           "Ambient Temperature: An Operando X-ray Absorption Spectroscopy Study', Angew. Chem. Int. Ed. 2023, 62, e202214032, "
           "doi:10.1002/anie.202214032; Zenodo doi:10.5281/zenodo.8041702")
    for p in sorted(base.glob("*.dat")):
        text = read_text(p)
        arr = numeric_block(text)
        if arr.size == 0:
            continue
        e = arr[:, 0]
        el = "Pt" if np.nanmedian(e) > 10000 else "Fe"
        edge = "L3" if el == "Pt" else "K"
        nsp = arr.shape[1] - 1
        mat = "Pt-Fe/SiO2" if "SiO2" in p.name else ("Pt-Fe/Al2O3" if "Al2O3" in p.name else "Pt-Fe oxide catalyst")
        mode = "fluorescence" if "fluo" in p.name else ("transmission" if "trans" in p.name else "QEXAFS ('Both' = fluorescence+transmission matrix as deposited)")
        cond = "PROX (CO+O2+H2) operando" if "PROX" in p.name else ("H2 treatment/TPR" if "H2" in p.name else "")
        rows.append(row(item_id="z8041702_" + safe(p.stem), source="zenodo_8041702_PtFeOx_operando_XAS", source_record_url=url,
                        file_path=rel(p), licence=CCBY, technique="xas", edge_or_wavelength=edge_label(el, edge), x_kind="energy_eV",
                        material=f"{mat} ({cond})" if cond else mat, composition="Pt-Fe-O", sample_type="catalyst",
                        notes=xas_notes(mode, e, el, edge, (f"TIME-RESOLVED MATRIX: column 'E' + {nsp} spectra (columns 0..{nsp-1}), operando QEXAFS"
                                                           if nsp > 1 else "single averaged spectrum") + f"; file name '{p.name}'"),
                        citation=cit))
    return rows


def src_athena_derived(srcdir, source, url, licence, cit, matfun):
    rows = []
    seen = {}
    for p in sorted((ROOT / srcdir / "derived").glob("*.dat")):
        text = read_text(p)
        arr = numeric_block(text)
        if arr.size == 0:
            continue
        e, mu = arr[:, 0], arr[:, 1]
        ok = e > 1000
        e, mu = e[ok], mu[ok]
        key = (len(e), round(float(e[0]), 2), round(float(np.nansum(mu)), 4))
        lab = re.search(r"# label: (.*)", text)
        lab = lab.group(1).strip() if lab else p.stem
        if key in seen:
            continue
        seen[key] = p.name
        el, edge = edge_from_e(e, ["Pt"])
        mat, comp, st = matfun(lab)
        rows.append(row(item_id=f"{source}_{safe(p.stem)}", source=source, source_record_url=url, file_path=rel(p), licence=licence,
                        technique="xas", edge_or_wavelength=edge_label(el, edge), x_kind="energy_eV", material=mat, composition=comp,
                        sample_type=st, notes=xas_notes("", e, el, edge, f"extracted from Athena project (label '{lab}') by text parsing; "
                                                        "columns x=energy eV, y=mu (as stored in project)"),
                        citation=cit))
    return rows


def ucl_mat(lab):
    l = lab.lower()
    if l.strip().startswith("ref"):
        return ("Pt foil (reference channel measured simultaneously)", "Pt", "foil")
    if "ptsn" in l:
        return (f"Pt-Sn/MFI zeolite catalyst ({lab.strip()})", "Pt-Sn", "catalyst")
    if "zsm5" in l:
        return (f"Pt/ZSM-5 catalyst ({lab.strip()})", "Pt", "catalyst")
    return (f"Pt/silicalite-1 (MFI) catalyst ({lab.strip()})", "Pt", "catalyst")


def cardiff_mat(lab):
    l = lab.lower()
    if l.strip().startswith("ref"):
        return ("Pt foil (reference channel measured simultaneously)", "Pt", "foil")
    if "pt foil" in l or l == "pt_foil":
        return ("Pt foil", "Pt", "foil")
    if "pto2" in l:
        return ("PtO2 reference", "PtO2", "compound")
    if re.match(r"^\w{4}$", l.strip()):
        return (f"unlabelled Athena group '{lab}' (Pt/Al2O3 series)", "Pt", "catalyst")
    return (f"Pt nanoparticles on Al2O3 ({lab.strip()})", "Pt", "catalyst")


def src_ucl():
    return src_athena_derived("ucl_rdr_19063886_PtSn_zeolite_XAFS", "ucl_rdr_19063886", "https://doi.org/10.5522/04/19063886",
                              "CC0-1.0", "UCL Research Data Repository, 'XAFS data @ Pt and Sn edges for samples in Structure-Activity "
                              "Relationships in Highly Active Platinum-Tin MFI-type Zeolite Catalysts for Propane Dehydrogenation', "
                              "doi:10.5522/04/19063886", ucl_mat)


def src_cardiff_xas():
    return src_athena_derived("cardiff_27052390_PtNP_glycerol_APR", "cardiff_27052390", "https://doi.org/10.17035/d.2018.0054109030",
                              CCBY, "Cardiff University dataset 'Directed Aqueous-Phase Reforming of Glycerol Through Tailored Platinum "
                              "Nanoparticles', doi:10.17035/d.2018.0054109030; Appl. Catal. B (2018) doi:10.1016/j.apcatb.2018.07.008", cardiff_mat)


def split_columns_csv(p, outdir, source, url, licence, cit, matfun, mode="", el_pref=None, x_kind="energy_eV", skip=()):
    """CSV with first column energy (or k) and one column per spectrum."""
    rows = []
    text = read_text(p).lstrip("\ufeff")
    lines = text.splitlines()
    head = lines[0].split(",")
    arr = numeric_block("\n".join(lines[1:]))
    x = arr[:, 0]
    for j, lab in enumerate(head[1:], start=1):
        if lab in skip or j >= arr.shape[1]:
            continue
        y = arr[:, j]
        ok = np.isfinite(y) & np.isfinite(x)
        dest = Path(outdir) / f"{Path(p).stem}__{safe(lab)}.dat"
        write_two_col(dest, x[ok], [y[ok]], [f"derived from {rel(p)} column '{lab}'"], [head[0], lab])
        el, edge = edge_from_e(x, el_pref) if x_kind == "energy_eV" else ("Pt", "L3")
        mat, comp, st = matfun(lab)
        rows.append(row(item_id=f"{source}_{safe(Path(p).stem)}_{safe(lab)}", source=source, source_record_url=url,
                        file_path=rel(dest), licence=licence, technique="xas", edge_or_wavelength=edge_label(el, edge),
                        x_kind=x_kind, material=mat, composition=comp, sample_type=st,
                        notes=(xas_notes(mode, x[ok], el, edge, f"column '{lab}' of {Path(p).name}") if x_kind == "energy_eV"
                               else f"chi(k) column '{lab}' of {Path(p).name}; npts={ok.sum()}; k={x.min():.2f}-{x.max():.2f}"),
                        citation=cit))
    return rows


def wright_mat(lab):
    refs = {"Pt(O2C5H7)2": ("Pt(acac)2 reference", "Pt(C5H7O2)2", "compound"), "PtCl2": ("PtCl2 reference", "PtCl2", "compound"),
            "PtCl4(NH4)2": ("(NH4)2PtCl4 reference", "(NH4)2PtCl4", "compound"),
            "PtCl6(NH4)2": ("(NH4)2PtCl6 reference", "(NH4)2PtCl6", "compound"), "PtO2": ("PtO2 reference", "PtO2", "compound")}
    if lab in refs:
        return refs[lab]
    host = "goethite" if lab.upper().startswith("G") or lab.startswith("hG") else "hematite"
    return (f"Pt(II) adsorbed on {host} ({lab})", "Pt on FeOOH/Fe2O3", "other")


def src_wright():
    rows = []
    b1 = ROOT / "figshare_30323788_Pt_Pd_adsorption_XAFS"
    rows += split_columns_csv(b1 / "Wright2025_PtXAFS_data.csv", b1 / "derived", "figshare_30323788",
                              "https://doi.org/10.6084/m9.figshare.30323788", CCBY,
                              "Wright E.G., Wang I., Fang Y., Flynn E.D., Catalano J.G. (2025) Data for: Climatic Imprint on "
                              "Interfacially-Controlled Platinum-Palladium Resources, figshare, doi:10.6084/m9.figshare.30323788",
                              wright_mat, mode="", el_pref=["Pt"])
    b2 = ROOT / "mendeley_9bhtr2w54w_Pt_adsorbed_minerals_XAS"
    cit2 = ("Wright E.G., Loza F.M., Flynn E.D., Catalano J.G. (2026) Data for: Mineral-specific ligand and redox dynamics of "
            "adsorbed platinum, Mendeley Data, doi:10.17632/9bhtr2w54w.1")
    for fn, tag in (("Wright2026_XANES_data.csv", "XANES"), ("Wright2026_EXAFS_data.csv", "EXAFS")):
        rr = split_columns_csv(b2 / fn, b2 / "derived", "mendeley_9bhtr2w54w", "https://doi.org/10.17632/9bhtr2w54w.1", CCBY,
                               cit2, wright_mat, el_pref=["Pt"])
        for r in rr:
            r["notes"] += f"; from {tag} file (XANES and EXAFS files contain the same samples, possibly differently processed)"
        rows += rr
    return rows


def src_pt_fluids():
    rows = []
    b = ROOT / "mendeley_w8bxyj33hr_Pt_fluids_XAS/extracted"
    cit = ("Tagirov B. et al. (2020) Data for: Platinum transport in chloride-bearing fluids and melts: insights from in situ "
           "X-ray absorption spectroscopy and thermodynamic modeling, Mendeley Data, doi:10.17632/w8bxyj33hr.1")
    for p in sorted(b.glob("*.xmu")):
        arr = numeric_block(read_text(p))
        e = arr[:, 0]
        t = re.search(r"_(\d+)C", p.name)
        tK = f"{int(t.group(1)) + 273.15:.2f}" if t else ""
        rows.append(row(item_id="mendeley_w8bxyj33hr_" + safe(p.stem), source="mendeley_w8bxyj33hr", source_record_url="https://doi.org/10.17632/w8bxyj33hr.1",
                        file_path=rel(p), licence=CCBY, technique="xas", edge_or_wavelength="Pt L3", x_kind="energy_eV",
                        material=f"Pt in chloride-bearing fluid/melt, in situ at {t.group(1) if t else '?'} C (merged scans)",
                        composition="Pt-Cl fluid", sample_type="other", temperature_K=tK,
                        notes=xas_notes("", e, "Pt", "L3", f"{arr.shape[1]} columns; high-T/high-P cell"), citation=cit))
    return rows


def src_alpha_pto2():
    rows = []
    b = ROOT / "zenodo_21874842_alphaPtO2_FeMn_HERFD"
    x = b / "Supplementary_Dataset-GCA.xlsx"
    sh = xlsx_read.read_sheets(x, only={"Dataset S7", "Dataset S8"})
    cit = ("Datasets for: Platinum occurs as alpha-PtO2 nanoparticles in deep-sea ferromanganese deposits (2026), Zenodo, "
           "doi:10.5281/zenodo.21874842")
    s7 = sh["Dataset S7"]
    names = s7[1]
    for j in range(0, len(names), 3):
        lab = names[j]
        if not lab:
            continue
        e, y = [], []
        for r in s7[3:]:
            try:
                e.append(float(r[j])); y.append(float(r[j + 1]))
            except (ValueError, IndexError):
                continue
        e, y = np.array(e), np.array(y)
        if e.size < 10:
            continue
        dest = b / "derived" / f"S7_PtL3_HERFD_XANES__{safe(lab)}.dat"
        write_two_col(dest, e, [y], [f"derived from {rel(x)} sheet 'Dataset S7' column '{lab}'"], ["energy_eV", "intensity"])
        ref = lab in ("PtCl2", "PtCl4", "α-PtO2")
        rows.append(row(item_id="z21874842_S7_" + safe(lab), source="zenodo_21874842_alphaPtO2", source_record_url="https://doi.org/10.5281/zenodo.21874842",
                        file_path=rel(dest), licence=CCBY, technique="xas", edge_or_wavelength="Pt L3", x_kind="energy_eV",
                        material=(f"{lab} reference" if ref else f"Pt-rich layer of FeMn crust/nodule {lab}"),
                        composition={"PtCl2": "PtCl2", "PtCl4": "PtCl4", "α-PtO2": "PtO2"}.get(lab, "Pt in Fe-Mn oxide"),
                        sample_type="compound" if ref else "other",
                        notes=xas_notes("HERFD-XANES", e, "Pt", "L3", "XANES range only"), citation=cit))
    s8 = sh["Dataset S8"]
    names = s8[1]
    for j, lab in enumerate(names):
        if not lab or s8[2][j] != "k":
            continue
        k, chi = [], []
        for r in s8[3:]:
            try:
                k.append(float(r[j])); chi.append(float(r[j + 1]))
            except (ValueError, IndexError):
                continue
        if len(k) < 10:
            continue
        dest = b / "derived" / f"S8_PtL3_EXAFS_chik__{safe(lab)}.dat"
        write_two_col(dest, np.array(k), [np.array(chi)], [f"derived from {rel(x)} sheet 'Dataset S8' column group '{lab}' (k, chi)"], ["k_inv_A", "chi"])
        rows.append(row(item_id="z21874842_S8_" + safe(lab), source="zenodo_21874842_alphaPtO2", source_record_url="https://doi.org/10.5281/zenodo.21874842",
                        file_path=rel(dest), licence=CCBY, technique="xas", edge_or_wavelength="Pt L3", x_kind="k",
                        material=(f"{lab} reference" if "PtO2" in lab else f"Pt in FeMn crust/nodule: {lab.strip()}"),
                        composition="PtO2" if "PtO2" in lab else "Pt in Fe-Mn oxide", sample_type="compound" if "PtO2" in lab else "other",
                        notes=f"EXAFS chi(k) (unweighted 'chi' column; file also has k-weighted columns); npts={len(k)}; kmax={max(k):.1f}",
                        citation=cit))
    return rows


def src_ni3ga():
    rows = []
    b = ROOT / "zenodo_16948020_Ni3Ga_NP/extracted"
    cit = ("Data for 'Structural Dynamics Behind the Formation of alpha'-Ni3Ga Alloy Nanoparticles from a Ni-Ga Phyllosilicate "
           "Dispersed on Silica Using X-ray Probes', Zenodo doi:10.5281/zenodo.16948020 (article doi:10.1021/acs.chemmater.5c01040)")
    url = "https://doi.org/10.5281/zenodo.16948020"
    for p in sorted(b.rglob("*.dat")):
        if "lab_XRD" in p.name or "_FT_" in p.name or "time_vs_temperature" in p.name or "scan_vs_temperature" in p.name or "Main_text/Figure3/All_EXAFS" in p.as_posix():
            continue
        text = read_text(p)
        lines = text.splitlines()
        head = lines[0].split("\t")
        arr_rows = []
        for ln in lines[1:]:
            parts = ln.split("\t")
            try:
                arr_rows.append([float(x) if x.strip() else np.nan for x in parts])
            except ValueError:
                continue
        if not arr_rows:
            continue
        ncol = max(len(r) for r in arr_rows)
        arr = np.array([r + [np.nan] * (ncol - len(r)) for r in arr_rows])
        is_k = head[0].strip().lower() in ("k",) or "k3EXAFS_k" in p.name or "_k3EXAFS" in p.name
        # column pairs (x, y) separated by empty columns
        pairs = []
        j = 0
        while j < len(head) - 1:
            if head[j].strip() and head[j + 1].strip():
                pairs.append((j, j + 1))
                j += 2
            else:
                j += 1
        if len(pairs) > 12:   # in-situ matrix: keep file as one item
            e = arr[:, 0]
            el, edge = ("Ni", "K") if "Ni" in p.name and "Ga_K" not in p.name else ("Ga", "K")
            rows.append(row(item_id="z16948020_" + safe(p.stem), source="zenodo_16948020_Ni3Ga", source_record_url=url, file_path=rel(p),
                            licence=CCBY, technique="xas", edge_or_wavelength=edge_label(el, edge), x_kind="energy_eV",
                            material="NiGa/SiO2 during H2 reduction (in situ)", composition="Ni-Ga/SiO2", sample_type="catalyst",
                            notes=xas_notes("in situ", e[np.isfinite(e)], el, edge, f"TIME-RESOLVED MATRIX with {len(pairs)} column pairs"),
                            citation=cit))
            continue
        for (a, c) in pairs:
            lab = head[c].strip()
            x, y = arr[:, a], arr[:, c]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 10:
                continue
            x, y = x[ok], y[ok]
            if is_k or head[a].strip().lower() == "k":
                el = "Ga" if "Ga_K" in p.name or "Ga_SiO2" in p.name else "Ni"
                xk, ek = "k", "K"
            else:
                el, ek = edge_from_e(x, ["Ni", "Ga"])
                xk = "energy_eV"
            dest = p.parent / "derived" / f"{p.stem}__{safe(lab)}.dat"
            write_two_col(dest, x, [y], [f"derived from {rel(p)} column '{lab}'"], ["k_inv_A" if xk == "k" else "energy_eV", lab])
            mat = lab
            comp = {"NiO": "NiO", "Ni-foil": "Ni", "beta-Ga2O3": "Ga2O3"}.get(lab, "Ni-Ga" if "NiGa" in lab or "Ni3Ga" in lab else ("Ni/SiO2" if lab.startswith("Ni_") else ("Ga/SiO2" if lab.startswith("Ga_") else "")))
            st = "foil" if "foil" in lab else ("compound" if lab in ("NiO", "beta-Ga2O3") else "catalyst")
            rows.append(row(item_id="z16948020_" + safe(p.stem) + "_" + safe(lab), source="zenodo_16948020_Ni3Ga", source_record_url=url,
                            file_path=rel(dest), licence=CCBY, technique="xas", edge_or_wavelength=edge_label(el, ek), x_kind=xk,
                            material=mat, composition=comp, sample_type=st,
                            notes=(xas_notes("", x, el, ek, f"column '{lab}' of {p.name}") if xk == "energy_eV"
                                   else f"k^3-weighted chi(k) column '{lab}' of {p.name}; npts={len(x)}; kmax={x.max():.1f}"),
                            citation=cit))
    return rows


def src_coo_herfd():
    rows = []
    b = ROOT / "zenodo_4746349_CoO_assemblies_HERFD_PXRD/HERFD_references"
    for p in sorted(b.glob("*.dat")):
        arr = numeric_block(read_text(p))
        if arr.size == 0:
            continue
        e = arr[:, 0]
        if np.nanmedian(e) < 100:
            e = e * 1000
        lab = p.stem.split("HERFD-XANES_")[-1]
        comp = "CoO" if "CoO" in lab else ("Co(C5H7O2)2" if "acac2" in lab else "Co(C5H7O2)3")
        rows.append(row(item_id="z4746349_" + safe(lab), source="zenodo_4746349_CoO_assemblies", source_record_url="https://doi.org/10.5281/zenodo.4746349",
                        file_path=rel(p), licence=CCBY, technique="xas", edge_or_wavelength="Co K", x_kind="energy_eV",
                        material=lab, composition=comp, sample_type="compound",
                        notes=xas_notes("HERFD-XANES (ESRF ID26, Si(531) analysers)", e, "Co", "K",
                                        f"{arr.shape[1]} columns; narrow XANES-only range; see header for column meaning"),
                        citation="Raw data for 'X-ray studies bridge the molecular and macro length scales during the emergence of CoO "
                                 "assemblies', Zenodo doi:10.5281/zenodo.4746349"))
    return rows
