"""XRD and PDF sources -> catalog rows (+ derived per-pattern files where a deposit packs several in one file)."""
import csv
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, r"D:/GAN/artifacts/data/real_raw_tools")
from catalog_common import ROOT, numeric_block, read_text, rel, write_two_col  # noqa: E402
from build_catalog_xas import row  # noqa: E402

CCBY = "CC-BY-4.0"


def safe(s):
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", str(s)).strip("_")[:80]


def wl_label(anode, wl):
    try:
        w = float(wl)
    except (TypeError, ValueError):
        w = None
    if anode:
        return f"{anode}Ka {w:.5g}" if w else f"{anode}Ka"
    if w:
        if 1.539 <= w <= 1.5425:
            return f"CuKa {w:.5g}"
        if 1.7885 <= w <= 1.7935:
            return f"CoKa {w:.5g}"
        if 0.7090 <= w <= 0.7115:
            return f"MoKa {w:.5g}"
        return f"{w:.5g} A"
    return "unknown"


def xrd_notes(tt, extra=""):
    parts = []
    if tt is not None and len(tt):
        parts.append(f"npts={len(tt)}; 2theta={np.nanmin(tt):.2f}-{np.nanmax(tt):.2f} deg; step~{np.nanmedian(np.diff(tt)):.4f}")
    if extra:
        parts.append(extra)
    return "; ".join(parts)


# ---------------------------------------------------------------- parsers for common lab formats
def parse_panalytical_csv(p):
    text = read_text(p).lstrip("\ufeff")
    hdr = {}
    lines = text.splitlines()
    i0 = 0
    for i, ln in enumerate(lines):
        if ln.startswith("[Scan points]"):
            i0 = i + 2
            break
        if "," in ln:
            k, v = ln.split(",", 1)
            hdr[k.strip()] = v.strip().strip('"')
    arr = numeric_block("\n".join(lines[i0:]))
    return hdr, arr


def parse_ras(p):
    text = read_text(p)
    hdr = dict(re.findall(r"^\*(\w+)\s+\"?([^\"\r\n]*)\"?", text, re.M))
    m = re.search(r"\*RAS_INT_START(.*?)\*RAS_INT_END", text, re.S)
    arr = numeric_block(m.group(1) if m else "")
    return hdr, arr


def parse_bruker_txt(p):
    text = read_text(p)
    hdr = {k: v.strip() for k, v in re.findall(r"^(\w+)=(.*)$", text, re.M)}
    i = text.find("[Data]")
    arr = numeric_block(text[i + 6:] if i >= 0 else text)
    return hdr, arr


# ---------------------------------------------------------------- XRD sources
def src_opxrd():
    rows = []
    idx = ROOT / "opxrd/_opxrd_index.csv"
    cit = ("Hollarek D., Friederich P. (2025) opXRD: Open Experimental Powder X-ray Diffraction Database, Zenodo "
           "doi:10.5281/zenodo.15298026 (arXiv:2503.05577)")
    with open(idx, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            w1, w2 = r["wavelength1"], r["wavelength2"]
            try:
                wv1 = float(w1) if w1 else None
            except ValueError:
                wv1 = None
            label = wl_label(None, wv1) if wv1 else "unknown"
            note_w = ""
            if wv1 and abs(wv1 - 1.54439) < 1e-4 and w2 and abs(float(w2) - 1.54056) < 1e-3:
                label = "CuKa 1.5406"
                note_w = "opXRD label lists primary=1.54439 / secondary=1.54056 (Cu Ka2/Ka1 swapped in label)"
            formulas = r["phase_formulas"].strip("; ")
            mat = formulas if formulas else "unlabelled (no phase label in opXRD)"
            notes = (f"institution={r['institution'][:60]}; contributor={r['contributor']}; original_format={r['original_format']}; "
                     f"npts={r['n_points']}; 2theta={r['tt_min']}-{r['tt_max']} deg; step~{r['step_med']}; "
                     f"lambda1={w1} lambda2={w2}; n_phases={r['n_phases']}; spacegroups={r['spacegroups'].strip('; ')}; "
                     f"JSON keys: two_theta_values, intensities, label, metadata" + (f"; {note_w}" if note_w else ""))
            rows.append(row(item_id="opxrd_" + safe(r["rel_path"].replace(".json", "")), source="opxrd",
                            source_record_url="https://doi.org/10.5281/zenodo.15298026",
                            file_path="opxrd/extracted/" + r["rel_path"], licence=CCBY, technique="xrd",
                            edge_or_wavelength=label, x_kind="two_theta_deg", material=mat,
                            composition=formulas.split(";")[0].strip() if formulas else "",
                            sample_type="other", temperature_K=r["temp_K"] if r["temp_K"] not in ("None", "") else "",
                            notes=notes, citation=cit))
    return rows


def src_htpem_xrd():
    rows = []
    base = ROOT / "zenodo_17131699_PtCo_PtNi_HTPEM/extracted"
    cit = ("Buriánek J.D., Bouzek K., Paušová Š. (2025) Dataset for 'Activity and Degradation of Pt-Co and Pt-Ni Alloy Catalysts "
           "for Application in High-temperature PEM Fuel Cells', Zenodo doi:10.5281/zenodo.17131699")
    code = {"GO": ("Pt-Co/reduced graphene oxide", "PtCo"), "RGO": ("Pt-Co/reduced graphene oxide", "PtCo"),
            "HIS": ("Pt/C (commercial)", "Pt"), "KET": ("Pt-Co/Ketjen Black", "PtCo"), "VUL": ("Pt-Co/Ketjen Black (Vulcan label)", "PtCo"),
            "NI": ("Pt-Ni/Ketjen Black", "PtNi"), "PT_C": ("Pt/C (commercial)", "Pt"), "PT-CO_C": ("Pt-Co/C", "PtCo"),
            "PT-CO_GO": ("Pt-Co/rGO", "PtCo"), "PT-NI_C": ("Pt-Ni/C", "PtNi"), "PT_CO-C": ("Pt-Co/C", "PtCo"),
            "PT_CO-RGO": ("Pt-Co/rGO", "PtCo"), "PT_NI-C": ("Pt-Ni/C", "PtNi")}
    for p in sorted(base.glob("VZ2_002_021_VSCHT_D_0*_v1.csv")):
        num = int(re.search(r"_D_(\d+)_", p.name).group(1))
        if not (309 <= num <= 324 or 414 <= num <= 456):
            continue
        hdr, arr = parse_panalytical_csv(p)
        sid = hdr.get("Sample identification", "")
        key = None
        for k in sorted(code, key=len, reverse=True):
            if sid.upper().replace(" ", "").startswith(k):
                key = k
                break
        mat, comp = code.get(key, (sid, ""))
        series = ("pristine (0) / leached 24 h in H3PO4 180 C (1-3)" if num <= 324 else
                  ("leached 1-72 h in 98 wt% H3PO4 at 180 C" if num <= 433 else "post-mortem GDE after in situ XAS"))
        rows.append(row(item_id=f"z17131699_xrd_{num}", source="zenodo_17131699_PtCo_PtNi_HTPEM",
                        source_record_url="https://doi.org/10.5281/zenodo.17131699", file_path=rel(p), licence=CCBY,
                        technique="xrd", edge_or_wavelength=wl_label(hdr.get("Anode material", ""), hdr.get("K-Alpha1 wavelength")),
                        x_kind="two_theta_deg", material=f"{mat} [{sid}]", composition=comp, sample_type="catalyst",
                        notes=xrd_notes(arr[:, 0], f"PANalytical X'Pert PRO, no monochromator (Ka1+Ka2, Ka2/Ka1=0.5); series: {series}; "
                                        "columns: Angle(2theta), TimePerStep, Intensity(counts), ESD after '[Scan points]'"),
                        citation=cit))
    return rows


def src_small_xrd():
    rows = []
    # Zenodo 15310405 (Pt/activated carbon, PANalytical CSV)
    b = ROOT / "zenodo_15310405_PtAC_hydrochlorination/extracted/XRD"
    for p in sorted(b.glob("*.csv")):
        hdr, arr = parse_panalytical_csv(p)
        name = p.stem
        comp = "Pt/C" if "Pt" in name else "C"
        rows.append(row(item_id="z15310405_" + safe(name), source="zenodo_15310405_PtAC", source_record_url="https://doi.org/10.5281/zenodo.15310405",
                        file_path=rel(p), licence=CCBY, technique="xrd", edge_or_wavelength=wl_label(hdr.get("Anode material"), hdr.get("K-Alpha1 wavelength")),
                        x_kind="two_theta_deg", material=("Pt nanoparticles on activated carbon" if "PtNP" in name else ("Pt single atoms on activated carbon" if "PtSA" in name else "activated carbon support")) + f" ({name})",
                        composition=comp, sample_type="catalyst" if "Pt" in name else "other",
                        notes=xrd_notes(arr[:, 0], "PANalytical X'Pert PRO, Ka1+Ka2; columns Angle, Intensity"),
                        citation="Giulimondi V. et al. (2025) Dataset for 'Influence of Hydrogen and Oxygen Impurities on Platinum-Catalyzed "
                                 "Acetylene Hydrochlorination', Zenodo doi:10.5281/zenodo.15310405"))
    # Zenodo 4952884 (Pt-Pd dendrimer NPs, PANalytical CSV, CC0)
    b = ROOT / "zenodo_4952884_PtPd_dendrimer"
    for p in sorted(b.glob("*.csv")):
        hdr, arr = parse_panalytical_csv(p)
        rows.append(row(item_id="z4952884_" + safe(p.stem), source="zenodo_4952884_PtPd_dendrimer", source_record_url="https://doi.org/10.5281/zenodo.4952884",
                        file_path=rel(p), licence="CC0-1.0", technique="xrd", edge_or_wavelength=wl_label(hdr.get("Anode material"), hdr.get("K-Alpha1 wavelength")),
                        x_kind="two_theta_deg", material=f"dendrimer-stabilised Pt-Pd bimetallic nanoparticles ({p.stem})",
                        composition="Pd7Pt3" if "PD7PT3" in p.stem else "Pt-Pd (ratio not given in file name)", sample_type="nanoparticle",
                        notes=xrd_notes(arr[:, 0], "PANalytical CSV, Ka1+Ka2; sample-to-composition mapping only explicit for PD7PT3 files"),
                        citation="Jin Z. et al. (2017) Data from: Synthesis and hydrogenation application of Pt-Pd bimetallic nanocatalysts "
                                 "stabilized by macrocycle-modified dendrimer, Zenodo doi:10.5281/zenodo.4952884"))
    # Zenodo 14186519 (GO-PtNP gamma; .DAT 2-col, wavelength not stated)
    b = ROOT / "zenodo_14186519_GO_PtNP_gamma/extracted"
    for p in sorted(b.glob("*.DAT")):
        arr = numeric_block("\n".join(read_text(p).splitlines()[1:]))
        rows.append(row(item_id="z14186519_" + safe(p.stem), source="zenodo_14186519_GO_PtNP", source_record_url="https://doi.org/10.5281/zenodo.14186519",
                        file_path=rel(p), licence=CCBY, technique="xrd", edge_or_wavelength="unknown (not stated; likely CuKa)", x_kind="two_theta_deg",
                        material=("graphene oxide" if p.stem == "GO" else f"Pt nanoparticles on graphene oxide, gamma-irradiated {p.stem}"),
                        composition="C" if p.stem == "GO" else "Pt/GO", sample_type="other" if p.stem == "GO" else "catalyst",
                        notes=xrd_notes(arr[:, 0], "first line = start step end label; then 2theta, counts"),
                        citation="Kepic D. et al. (2024) Synthesis of Pt nanoparticles on GO under low-dose gamma irradiation, Zenodo doi:10.5281/zenodo.14186519"))
    # Zenodo 21028650 / 21028904 (Graphenea GO-Pt; Rigaku MiniFlex600; CC0)
    for rid, pct in (("21028650", "10"), ("21028904", "20")):
        b = ROOT / f"zenodo_{rid}_GOPt{pct}_XRD"
        for p in sorted(b.glob("*.csv")):
            arr = numeric_block("\n".join(read_text(p).splitlines()[2:]))
            rows.append(row(item_id=f"z{rid}_GOPt{pct}", source=f"zenodo_{rid}_GOPt{pct}", source_record_url=f"https://doi.org/10.5281/zenodo.{rid}",
                            file_path=rel(p), licence="CC0-1.0", technique="xrd", edge_or_wavelength="CuKa (assumed: Rigaku MiniFlex600, not stated)",
                            x_kind="two_theta_deg", material=f"Pt nanoparticles ({pct} wt%) on graphene oxide", composition="Pt/GO", sample_type="catalyst",
                            notes=xrd_notes(arr[:, 0], "Rigaku MiniFlex600, 7-70 deg at 1 deg/min; intensity in cps"),
                            citation=f"Kuhnt T., Alonso B. (2026) XRD_ANALYSIS_GOPt-{pct}wt%_WP2_T2.2_GRA, Zenodo doi:10.5281/zenodo.{rid}"))
    # DataverseNL 7M9Q6K (AuCu-core Pt-shell NPs; Co anode)
    b = ROOT / "dataversenl_7M9Q6K_AuCu_Pt_coreshell_XRD"
    for p in sorted(b.glob("*.xy")):
        text = read_text(p)
        arr = numeric_block("\n".join(text.splitlines()[1:]))
        an = re.search(r'Anode: "(\w+)"', text)
        lab = p.stem.replace("Fig5a_", "").replace("_XRD", "")
        rows.append(row(item_id="dvnl_7M9Q6K_" + safe(lab), source="dataversenl_7M9Q6K", source_record_url="https://doi.org/10.34894/7M9Q6K",
                        file_path=rel(p), licence=CCBY, technique="xrd", edge_or_wavelength=wl_label(an.group(1) if an else "", 1.78897 if an and an.group(1) == "Co" else None),
                        x_kind="two_theta_deg", material=f"AuCu-core Pt-shell nanoparticles ({lab})", composition=lab.replace("_", "@"),
                        sample_type="nanoparticle", notes=xrd_notes(arr[:, 0], "Coupled TwoTheta/Theta, Co anode (lambda Ka1 1.78897 A assumed); lattice parameters in Fig5b csv"),
                        citation="Synthetic control over lattice strain in trimetallic AuCu-core Pt-shell nanoparticles (2025), DataverseNL doi:10.34894/7M9Q6K"))
    # Zenodo 17886350 (Pt on MoS2/WS2/Mo-W-S; Bruker RAW4 text)
    b = ROOT / "zenodo_17886350_Pt_on_MoS2_WS2_XRD"
    for p in sorted(b.glob("Fig_*.txt")):
        hdr, arr = parse_bruker_txt(p)
        lab = p.stem
        rows.append(row(item_id="z17886350_" + safe(lab), source="zenodo_17886350_Pt_TMS", source_record_url="https://doi.org/10.5281/zenodo.17886350",
                        file_path=rel(p), licence=CCBY, technique="xrd", edge_or_wavelength=wl_label(hdr.get("Anode", ""), hdr.get("Alpha1")),
                        x_kind="two_theta_deg", material=(f"Pt supported on transition-metal sulfide ({lab}; MW/WIR = Pt deposition method)" if ("_MW" in lab or "_WIR" in lab) else f"bare sulfide support, no platinum ({lab})"),
                        composition="Pt/MoS2-WS2" if ("_MW" in lab or "_WIR" in lab) else "MoS2/WS2 support",
                        sample_type="catalyst" if ("_MW" in lab or "_WIR" in lab) else "other",
                        notes=xrd_notes(arr[:, 0], "Bruker RAW4 text export: header key=value, data after [Data] (Angle, PSD)"),
                        citation="Dataset for Hydrogenation of Cinnamaldehyde by Platinum Supported 2D and 3D Transition Metal Sulfides, Zenodo doi:10.5281/zenodo.17886350"))
    # Cardiff (Rigaku .ras)
    b = ROOT / "cardiff_27052390_PtNP_glycerol_APR/extracted/Data/XRD - Fig 2_14_S1"
    for p in sorted(b.glob("*.ras")):
        hdr, arr = parse_ras(p)
        rows.append(row(item_id="cardiff_27052390_xrd_" + safe(p.stem), source="cardiff_27052390", source_record_url="https://doi.org/10.17035/d.2018.0054109030",
                        file_path=rel(p), licence=CCBY, technique="xrd",
                        edge_or_wavelength=wl_label(hdr.get("HW_XG_TARGET_NAME", ""), hdr.get("HW_XG_WAVE_LENGTH_ALPHA1")),
                        x_kind="two_theta_deg", material=f"Pt nanoparticles on Al2O3 ({p.stem})", composition="Pt/Al2O3", sample_type="catalyst",
                        notes=xrd_notes(arr[:, 0], "Rigaku RAS: header '*KEY \"value\"', data between *RAS_INT_START/*RAS_INT_END (2theta, counts, attenuation); .asc/.raw copies also in folder"),
                        citation="Cardiff University dataset 'Directed Aqueous-Phase Reforming of Glycerol Through Tailored Platinum Nanoparticles', doi:10.17035/d.2018.0054109030"))
    # Wright mineral XRD (column pairs)
    for b, fn, src, url, cit in (
            (ROOT / "figshare_30323788_Pt_Pd_adsorption_XAFS", "Wright2025_XRD_data.csv", "figshare_30323788", "https://doi.org/10.6084/m9.figshare.30323788",
             "Wright E.G. et al. (2025) Data for: Climatic Imprint on Interfacially-Controlled Platinum-Palladium Resources, figshare doi:10.6084/m9.figshare.30323788"),
            (ROOT / "mendeley_9bhtr2w54w_Pt_adsorbed_minerals_XAS", "Wright2026_XRD_data.csv", "mendeley_9bhtr2w54w", "https://doi.org/10.17632/9bhtr2w54w.1",
             "Wright E.G. et al. (2026) Data for: Mineral-specific ligand and redox dynamics of adsorbed platinum, Mendeley Data doi:10.17632/9bhtr2w54w.1")):
        p = b / fn
        lines = read_text(p).lstrip("\ufeff").splitlines()
        head = lines[0].split(",")
        rows_num = []
        for ln in lines[1:]:
            try:
                rows_num.append([float(x) if x.strip() else np.nan for x in ln.split(",")])
            except ValueError:
                continue
        ncol = max(len(r) for r in rows_num)
        arr = np.array([r + [np.nan] * (ncol - len(r)) for r in rows_num])
        for j in range(0, len(head) - 1, 2):
            lab = head[j].replace("_twotheta_degrees", "")
            x, y = arr[:, j], arr[:, j + 1]
            ok = np.isfinite(x) & np.isfinite(y)
            dest = b / "derived_xrd" / f"{p.stem}__{safe(lab)}.dat"
            write_two_col(dest, x[ok], [y[ok]], [f"derived from {rel(p)} columns '{head[j]}','{head[j+1]}'"], ["two_theta_deg", "counts"])
            mineral = "goethite" if lab.upper().startswith("G") or lab.startswith("hG") else "hematite"
            rows.append(row(item_id=f"{src}_xrd_{safe(lab)}", source=src, source_record_url=url, file_path=rel(dest), licence=CCBY,
                            technique="xrd", edge_or_wavelength="unknown (not stated in deposit)", x_kind="two_theta_deg",
                            material=f"synthetic {mineral} batch {lab}", composition="FeOOH" if mineral == "goethite" else "Fe2O3",
                            sample_type="other", notes=xrd_notes(x[ok], "collection conditions vary per batch (see article)"), citation=cit))
    # Ni3Ga lab XRD (in Q)
    b = ROOT / "zenodo_16948020_Ni3Ga_NP/extracted/2_SI"
    for p in sorted(b.glob("FS*_lab_XRD_*.dat")):
        lines = read_text(p).splitlines()
        head = lines[0].split("\t")
        data = []
        for ln in lines[2:]:
            try:
                data.append([float(x) if x.strip() else np.nan for x in ln.split("\t")])
            except ValueError:
                continue
        ncol = max(len(r) for r in data)
        arr = np.array([r + [np.nan] * (ncol - len(r)) for r in data])
        for j in range(len(head) - 1):
            if head[j].strip() == "Q" and head[j + 1].strip():
                lab = head[j + 1].strip()
                x, y = arr[:, j], arr[:, j + 1]
                ok = np.isfinite(x) & np.isfinite(y)
                dest = p.parent / "derived" / f"{p.stem}__{safe(lab)}.dat"
                write_two_col(dest, x[ok], [y[ok]], [f"derived from {rel(p)} column '{lab}' (x = Q in 1/A)"], ["q_inv_A", "intensity"])
                rows.append(row(item_id=f"z16948020_xrd_{safe(p.stem)}_{safe(lab)}", source="zenodo_16948020_Ni3Ga", source_record_url="https://doi.org/10.5281/zenodo.16948020",
                                file_path=rel(dest), licence=CCBY, technique="xrd", edge_or_wavelength="lab source (wavelength not stated; x given as Q)",
                                x_kind="q", material=lab, composition="Ni-Ga/SiO2" if "NiGa" in lab else ("SiO2" if lab == "SiO2" else lab),
                                sample_type="catalyst" if "DP" in lab else "other",
                                notes=f"lab XRD converted to Q by authors; npts={ok.sum()}; Q={x[ok].min():.2f}-{x[ok].max():.2f} 1/A",
                                citation="Data for 'Structural Dynamics Behind the Formation of alpha'-Ni3Ga Alloy Nanoparticles ...', Zenodo doi:10.5281/zenodo.16948020"))
    return rows


# ---------------------------------------------------------------- PDF / total scattering
def gr_header(p):
    text = read_text(p)[:6000]
    get = lambda k: (re.search(rf"^\s*{k}\s*=\s*(.*)$", text, re.M) or [None, ""])[1].strip()
    return {k: get(k) for k in ("wavelength", "composition", "qmax", "qmin", "qmaxinst", "rmax", "dataformat", "mode", "bgscale", "rpoly", "inputfile")}


def pdf_row(p, item_id, source, url, licence, cit, material, composition, sample_type, x_kind="r_A", extra="", wl=None):
    h = gr_header(p)
    arr = numeric_block(read_text(p))
    x = arr[:, 0] if arr.size else np.array([])
    rng = f"npts={len(x)}; x={x.min():.3f}-{x.max():.3f}" if len(x) else ""
    hn = "; ".join(f"{k}={v}" for k, v in h.items() if v)
    w = wl or (f"{h['wavelength']} A" if h.get("wavelength") else "unknown")
    return row(item_id=item_id, source=source, source_record_url=url, file_path=rel(p), licence=licence, technique="pdf",
               edge_or_wavelength=w, x_kind=x_kind, material=material, composition=composition or h.get("composition", ""),
               sample_type=sample_type, notes="; ".join(z for z in (rng, hn, extra) if z), citation=cit)


def src_pdf():
    rows = []
    # DeepStruc (Pt NP PDFs + Au144)
    b = ROOT / "github_DeepStruc"
    cit = ("Kjaer E.T.S., Anker A.S. et al. (2023) DeepStruc, Digital Discovery; experimental PDFs from github.com/EmilSkaaning/DeepStruc "
           "(data/PDFs_experimental), commit " + read_text(b / "COMMIT.txt").split()[0][:10])
    for p in sorted(b.glob("*.gr")):
        pt = p.stem.startswith("JQ")
        rows.append(pdf_row(p, "deepstruc_" + safe(p.stem), "github_DeepStruc", "https://github.com/EmilSkaaning/DeepStruc/tree/main/data/PDFs_experimental",
                            "Apache-2.0 (repository licence covering included data)", cit,
                            "colloidal Pt nanoparticles (surfactant-free synthesis, J. Quinson samples), DESY Sep 2017" if pt else "Au144(SR)60 clusters",
                            "Pt" if pt else "Au144(SR)60", "nanoparticle",
                            extra="G(r) from xPDFsuite/PDFgetX3 (header block then '#### start data'); background = ethylene glycol (JQ_EG)" if pt else "G(r)"))
    # scatterlab Au NPs (DanMAX) - G(r) and raw I(Q)
    b = ROOT / "github_scatterlab_AuNP_TSPDF/extracted/scatterlab-2026-tspdf-target-scattering-main/experiments/danmax20240084"
    cit = ("Anker A.S. et al. (2026) Autonomous Synthesis of Nanoparticles with Target Scattering Patterns (ScatterLab, DanMAX/MAX IV); "
           "github.com/AndySAnker/scatterlab-2026-tspdf-target-scattering (Apache-2.0); data also Zenodo doi:10.5281/zenodo.18427148 (CC-BY-4.0)")
    lic = "Apache-2.0 (repository); same experiment deposited on Zenodo 10.5281/zenodo.18427148 under CC-BY-4.0"
    url = "https://github.com/AndySAnker/scatterlab-2026-tspdf-target-scattering"
    for p in sorted((b / "plotting/Scattering_modelling/Rietveld_refinement/Iq_Gr_Rietveld_refinement/Gr").glob("*.gr")):
        rows.append(pdf_row(p, "scatterlab_gr_" + safe(p.stem), "github_scatterlab_AuNP", url, lic, cit,
                            "colloidal Au nanoparticles (self-driving-lab synthesis, one per experiment)", "Au", "nanoparticle",
                            extra="background-subtracted (bq_subtracted) G(r), 2 columns r, G"))
    for p in sorted((b / "plotting/Scattering_modelling/Rietveld_refinement/Iq_Gr_Rietveld_refinement/Iq").glob("*.xy")):
        kind = "blank (solvent/capillary)" if p.stem.endswith("_blank") else ("background-subtracted" if "bq_subtracted" in p.stem else "raw sample")
        rows.append(pdf_row(p, "scatterlab_iq_" + safe(p.stem), "github_scatterlab_AuNP", url, lic, cit,
                            f"colloidal Au nanoparticles - {kind} total scattering I(Q)" if "blank" not in p.stem else "solvent/capillary blank for Au NP synthesis",
                            "Au" if "blank" not in p.stem else "", "nanoparticle" if "blank" not in p.stem else "other", x_kind="q",
                            extra=f"I(Q) {kind}; 2 columns Q (1/A), I"))
    for sub in ("ClosedLoopExp_TargetDecahedron326_best41_71_Summation60sec", "ClosedLoopExp_TargetDecahedron326_best41_71_Summation1500sec"):
        for p in sorted((b / "data" / sub / "Gr").glob("*.gr")):
            rows.append(pdf_row(p, f"scatterlab_{'60s' if '60sec' in sub else '1500s'}_" + safe(p.stem), "github_scatterlab_AuNP", url, lic, cit,
                                "colloidal Au nanoparticles (best decahedral-target sample), repeated acquisitions", "Au", "nanoparticle",
                                extra=f"{'60 s' if '60sec' in sub else '1500 s'} summations of the same sample (useful for noise statistics)"))
    for p in sorted((b / "plotting/Supporting_Information/Washing_light_effect/Data-Postprocessed").glob("*.gr")):
        rows.append(pdf_row(p, "scatterlab_wash_" + safe(p.stem), "github_scatterlab_AuNP", url, lic, cit, "colloidal Au nanoparticles (SI washing/light test)", "Au", "nanoparticle"))
    for p in sorted((b / "plotting/Scattering_modelling/Rietveld_refinement/Si_with_backgroundsubtraction/Gr").glob("*.gr")):
        rows.append(pdf_row(p, "scatterlab_Si_" + safe(p.stem), "github_scatterlab_AuNP", url, lic, cit, "Si standard (DanMAX)", "Si", "other"))
    # diffpy.srfit examples
    b = ROOT / "github_diffpy_srfit_examples"
    cit = "diffpy.srfit documentation example data, github.com/diffpy/diffpy.srfit (docs/examples/data), commit " + read_text(b / "COMMIT.txt").split()[0][:10]
    lic = "DiffPy-CMI licence (BSD-like, redistribution permitted; repository licence covering example data)"
    meta = {"ni-q27r100-neutron.gr": ("Ni powder (commercial), neutron PDF, NPDF LANSCE, 300 K", "Ni", "foil", "neutron (TOF)", "300"),
            "ni-q27r60-xray.gr": ("Ni powder, x-ray PDF, APS 6-ID-D (MUCAT), 300 K", "Ni", "foil", "synchrotron x-ray (wavelength in header)", "300"),
            "si-q27r60-xray.gr": ("Si powder, x-ray PDF, APS 6-ID-D", "Si", "other", "synchrotron x-ray (wavelength in header)", "300"),
            "si90ni10-q27r60-xray.gr": ("Si90Ni10 two-phase mixture, x-ray PDF, APS 6-ID-D", "Si0.9Ni0.1 (mixture)", "other", "synchrotron x-ray (wavelength in header)", "300")}
    for p in sorted(b.glob("*.gr")):
        m = meta.get(p.name)
        r = pdf_row(p, "diffpy_" + safe(p.stem), "github_diffpy_srfit_examples", "https://github.com/diffpy/diffpy.srfit/tree/main/docs/examples/data",
                    lic, cit, m[0], m[1], m[2], extra="Qmax 27 A-1 (file name); PDFgetX2/PDFgetN header", wl=m[3])
        r["temperature_K"] = m[4]
        r["sample_type"] = "other" if m[2] == "foil" else m[2]
        rows.append(r)
    # Jensen-group experimental PDFs (ML-MotEx, ClusterFinder)
    b = ROOT / "github_jensen_group_exp_PDFs"
    for sub, url in (("ML-MotEx", "https://github.com/AndySAnker/ML-MotEx"), ("ClusterFinder", "https://github.com/AndySAnker/ClusterFinder")):
        for p in sorted((b / sub).glob("*.gr")):
            rows.append(pdf_row(p, f"{sub}_" + safe(p.stem), f"github_{sub}", url, "Apache-2.0 (repository licence covering included data)",
                                f"{sub} repository experimental data, {url}, commit {read_text(b / sub / 'COMMIT.txt').split()[0][:10]}",
                                f"experimental PDF {p.stem} (molecular clusters / polyoxometalates / oxide NPs)", "", "other"))
    # Zenodo PDF datasets (raw synchrotron patterns or G(r))
    z = [("zenodo_2593613_PDF_config_comparison_Ni_ID22_MSPD", "https://doi.org/10.5281/zenodo.2593613",
          "Comparative study of experimental configurations in synchrotron PDF (ESRF ID22/ID15A, ALBA MSPD), Zenodo doi:10.5281/zenodo.2593613"),
         ("zenodo_1255630_PDF_yeelimite_pastes_Ni_std", "https://doi.org/10.5281/zenodo.1255630",
          "Synchrotron PDF analyses of Ye'elimite-based pastes, Zenodo doi:10.5281/zenodo.1255630"),
         ("zenodo_890585_PDF_cement_gels_Ni_std", "https://doi.org/10.5281/zenodo.890585",
          "Synchrotron Radiation PDF Analysis of gels in cements, Zenodo doi:10.5281/zenodo.890585"),
         ("zenodo_14944751_lab_PDF_protocols_PDFgetX3", "https://doi.org/10.5281/zenodo.14944751",
          "Testing Protocols for Obtaining Reliable PDFs from Laboratory x-ray Sources Using PDFgetX3, Zenodo doi:10.5281/zenodo.14944751"),
         ("zenodo_15069353_DNA_Ag16Cl2_TS", "https://doi.org/10.5281/zenodo.15069353",
          "Supporting dataset: Structure of a DNA-stabilised Ag16Cl2 nanocluster in solution, Zenodo doi:10.5281/zenodo.15069353")]
    for d, url, cit in z:
        for p in sorted((ROOT / d).rglob("*")):
            if p.suffix.lower() not in (".dat", ".tth", ".xy", ".gr", ".fq", ".iq", ".sq", ".chi") or not p.is_file():
                continue
            nm = p.stem.lower()
            if p.suffix.lower() == ".gr":
                xk, kind = "r_A", "G(r)"
            elif p.suffix.lower() == ".fq":
                xk, kind = "q", "F(Q)"
            elif p.suffix.lower() in (".tth", ".dat") and "zenodo_14944751" not in d:
                xk, kind = "two_theta_deg", "raw synchrotron total-scattering pattern I(2theta) for PDF"
            elif "zenodo_15069353" in d:
                xk, kind = "q", "total scattering I(Q) (mean of frames)"
            else:
                xk, kind = ("two_theta_deg" if "zenodo_14944751" in d and p.suffix.lower() == ".xy" and "qscale" not in nm else "q"), "lab x-ray total scattering"
            arr = numeric_block(read_text(p))
            if arr.size == 0:
                continue
            xmed = float(np.nanmedian(arr[:, 0]))
            if xk == "two_theta_deg" and xmed > 200:
                xk = "q"
            mat = p.stem.replace("_", " ")
            comp = "Ni" if "nickel" in nm or nm.startswith("ni") else ("SiO2" if ("quartz" in nm or "silica" in nm) else
                    ("Ag16Cl2 (DNA-stabilised)" if "zenodo_15069353" in d else ""))
            st = "other"
            if "empty" in nm or "kapton" in nm or "capillary" in nm:
                mat = f"empty capillary / container background ({p.stem})"
                comp = ""
            rows.append(row(item_id=f"{d.split('_')[0]}_{d.split('_')[1]}_" + safe(p.stem) + "_" + p.suffix.lower().strip("."), source=d, source_record_url=url,
                            file_path=rel(p), licence=CCBY, technique="pdf", edge_or_wavelength="see source (synchrotron/lab; not in file)",
                            x_kind=xk, material=mat, composition=comp, sample_type=st,
                            notes=f"{kind}; npts={arr.shape[0]}; x={arr[:, 0].min():.3f}-{arr[:, 0].max():.3f}; {arr.shape[1]} columns", citation=cit))
    # ncomm gold NP 2016 (zip)
    zp = ROOT / "zenodo_56682_ncomm_goldnp_2016_Au144_PDF/ncomm-goldnp-2016-v3.0.zip"
    xdir = zp.parent / "extracted"
    if not xdir.exists():
        with zipfile.ZipFile(zp) as zf:
            for m in zf.infolist():
                if m.is_dir() or ".." in m.filename or m.filename.startswith("/"):
                    continue
                dest = xdir / Path(*Path(m.filename).parts)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(zf.read(m))
    for p in sorted(xdir.rglob("*")):
        if p.is_file() and p.suffix.lower() in (".gr", ".chi", ".fq", ".iq", ".sq", ".dat", ".xy"):
            xk = "r_A" if p.suffix.lower() == ".gr" else "q"
            arr = numeric_block(read_text(p))
            if arr.size == 0:
                continue
            rows.append(row(item_id="z56682_" + safe(p.stem) + "_" + p.suffix.lower().strip("."), source="zenodo_56682_ncomm_goldnp_2016", source_record_url="https://doi.org/10.5281/zenodo.56682",
                            file_path=rel(p), licence=CCBY, technique="pdf", edge_or_wavelength="see header/article (synchrotron)", x_kind=xk,
                            material="Au144(SR)60 clusters", composition="Au144(SR)60", sample_type="nanoparticle",
                            notes=f"npts={arr.shape[0]}; x={arr[:, 0].min():.3f}-{arr[:, 0].max():.3f}",
                            citation="Jensen K.M.O. et al. (2016) Polymorphism in magic-sized Au144(SR)60 clusters, Nat. Commun. 7, 11859; data Zenodo doi:10.5281/zenodo.56682"))
    # Ni3Ga PDFs / total scattering
    b = ROOT / "zenodo_16948020_Ni3Ga_NP/extracted"
    for p in sorted(b.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in (".gr", ".fq", ".iq", ".xye", ".qye"):
            continue
        xk = {".gr": "r_A", ".fq": "q", ".iq": "q", ".qye": "q", ".xye": "two_theta_deg"}[p.suffix.lower()]
        arr = numeric_block(read_text(p))
        if arr.size == 0:
            continue
        nm = p.stem
        comp = "Ni-Ga/SiO2" if "NiGa" in nm else ("Ni/SiO2" if "Ni_" in nm else ("Ga/SiO2" if "Ga_" in nm else ""))
        rows.append(row(item_id="z16948020_" + safe(rel(p).split("extracted/")[1]), source="zenodo_16948020_Ni3Ga", source_record_url="https://doi.org/10.5281/zenodo.16948020",
                        file_path=rel(p), licence=CCBY, technique="pdf", edge_or_wavelength="synchrotron (see .cfg / article)", x_kind=xk,
                        material=nm, composition=comp, sample_type="catalyst" if comp else "other",
                        notes=f"{p.suffix.lower()} file; npts={arr.shape[0]}; x={arr[:, 0].min():.3f}-{arr[:, 0].max():.3f}; PDFgetX3 .cfg files alongside",
                        citation="Data for 'Structural Dynamics Behind the Formation of alpha'-Ni3Ga Alloy Nanoparticles ...', Zenodo doi:10.5281/zenodo.16948020"))
    return rows
