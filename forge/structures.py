"""Phase 1a: reference bulk phases for Pt and Pt alloys from open databases.

Sources, in order: Crystallography Open Database (CC0), Materials Project (only if MP_API_KEY is set),
JARVIS-DFT 3D (CC BY 4.0), OQMD (CC BY 4.0). Candidates are classified by prototype (A1, L1_2, L1_0, L1_1)
and deduplicated with pymatgen's StructureMatcher. Experimental (COD, ambient) entries are preferred; a
DFT-only phase is rescaled isotropically by the same database's own a(Pt, exp)/a(Pt, DFT) ratio.

Outputs (artifacts/data/work/structures/): candidates.csv, reference_phases.json, cif/<id>.cif, structures.log
"""
import csv
import json
import logging
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import requests
from pymatgen.core import Lattice, Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .paths import STRUCT_DIR, load_env, sha256_bytes

LOG = logging.getLogger("structures")
CIF_DIR = STRUCT_DIR / "cif"
RAW_DIR = STRUCT_DIR / "raw"
LICENCES = {
    "COD": "CC0-1.0 (COD public-domain dedication)",
    "JARVIS": "CC-BY-4.0 (figshare 10.6084/m9.figshare.6815699)",
    "OQMD": "CC-BY-4.0",
    "MP": "CC-BY-4.0",
}
ALLOY_ELEMENTS = ["Co", "Ni", "Fe", "Cu"]
KX_TO_A = 1.002056          # pre-1947 lattice constants are in kX units
KX_YEAR = 1947
# entries that are not ambient single-phase reference measurements (pressure markers, T series, films, theory)
EXCLUDE_TITLE = re.compile(r"pressure|compress|thermal expansion|lattice expansion|temperat|theory of bonding|"
                           r"sputtered|evaporated", re.I)
# COD 5910049 (CuPt, Wyckoff 1931) is misread by the CIF parser (12-site Cu3Pt9); it is rebuilt as L1_1 from the
# reported pseudo-cubic rhombohedral cell (a, alpha)
COD_L11_REBUILD = {"5910049": ("Pt", "Cu")}

# Target systems. The ordered prototype is the experimentally observed one. PtCu orders in the CuPt-type
# L1_1 structure (alternating (111) planes), not L1_0, so L1_1 is used for PtCu (documented deviation).
SYSTEMS = {
    "Pt": dict(M=None, x=0.0, ordered=None),
    "Pt3Co": dict(M="Co", x=0.25, ordered="L1_2"),
    "PtCo": dict(M="Co", x=0.50, ordered="L1_0"),
    "Pt3Ni": dict(M="Ni", x=0.25, ordered="L1_2"),
    "PtNi": dict(M="Ni", x=0.50, ordered="L1_0"),
    "Pt3Fe": dict(M="Fe", x=0.25, ordered="L1_2"),
    "PtCu": dict(M="Cu", x=0.50, ordered="L1_1"),
}


# ----------------------------------------------------------------------------------------------- prototypes
def _fcc_int_sites(n=2):
    """Integer fcc coordinates (units of a/2) in an n x n x n conventional supercell."""
    pts = []
    for i in range(2 * n):
        for j in range(2 * n):
            for k in range(2 * n):
                if (i + j + k) % 2 == 0:
                    pts.append((i, j, k))
    return np.array(pts)


def prototype_structures():
    """Ideal prototypes with species A (=Pt-like) and B (=M-like) for anonymous matching."""
    protos = {}
    fcc = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    protos["A1"] = Structure(Lattice.cubic(1.0), ["Cu"] * 4, fcc)
    protos["L1_2"] = Structure(Lattice.cubic(1.0), ["Au", "Cu", "Cu", "Cu"], fcc)
    protos["L1_0"] = Structure(Lattice.tetragonal(1.0, 0.95), ["Au", "Au", "Cu", "Cu"], fcc)
    n = _fcc_int_sites(2)
    m = ((n[:, 0] + n[:, 1] + n[:, 2]) // 2) % 2
    protos["L1_1"] = Structure(Lattice.cubic(2.0), ["Au" if v == 0 else "Cu" for v in m], n / 4.0)
    return protos


def l11_from_rhombo(a, alpha_deg, A="Pt", B="Cu"):
    """L1_1 (CuPt-type) 2x2x2 pseudo-cubic supercell for a face-centred rhombohedral cell (a, alpha)."""
    ca = np.cos(np.radians(alpha_deg))
    n = _fcc_int_sites(2)
    m = ((n[:, 0] + n[:, 1] + n[:, 2]) // 2) % 2
    # lattice vectors of the deformed 2a cube: equal lengths 2a, equal angles alpha
    v1 = np.array([1.0, 0, 0])
    v2 = np.array([ca, np.sqrt(1 - ca ** 2), 0])
    cx = ca
    cy = (ca - ca * ca) / np.sqrt(1 - ca ** 2)
    v3 = np.array([cx, cy, np.sqrt(1 - cx ** 2 - cy ** 2)])
    lat = Lattice(np.stack([v1, v2, v3]) * 2 * a)
    return Structure(lat, [A if v == 0 else B for v in m], n / 4.0)


def title_temperature(title):
    m = re.search(r"T\s*=\s*(-?[0-9.]+)\s*([KC])", title or "")
    if not m:
        return None
    t = float(m.group(1))
    return t + 273.15 if m.group(2) == "C" else t


def classify(struct):
    """Return prototype label for an (ordered or site-disordered) structure, or None."""
    try:
        sga = SpacegroupAnalyzer(struct, symprec=0.05)
        sg = sga.get_space_group_number()
    except Exception:
        sg = None
    if not struct.is_ordered:
        if sg == 225 and len(struct.composition.elements) >= 1:
            return "A1", sg
        return None, sg
    sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True, scale=True, attempt_supercell=True)
    for name, proto in prototype_structures().items():
        if len(struct.composition.elements) != len(proto.composition.elements):
            continue
        # match fractional composition (A1 single element; ordered binaries AB3 / AB)
        if _round_comp(struct) != _round_comp(proto):
            continue
        if sm.fit_anonymous(struct, proto):
            return name, sg
    return None, sg


def _round_comp(s):
    return sorted(round(v, 6) for v in s.composition.fractional_composition.values())


def pseudo_cubic_params(struct, proto):
    """Pseudo-cubic (fcc-like) lattice description for the particle builder.

    Returns dict(a=..., c=..., alpha=..., a_mean=...) where for L1_0 a/c refer to the fcc-like conventional
    cell (a = a_t * sqrt(2) when given in the primitive tetragonal cell) and for L1_1 alpha is the angle of the
    face-centred rhombohedral (pseudo-cubic) cell. a_mean is the cube root of the volume per 4 atoms.
    """
    prim = SpacegroupAnalyzer(struct, symprec=0.05).get_conventional_standard_structure() if struct.is_ordered else struct
    vol_per_atom = struct.volume / len(struct) if struct.is_ordered else struct.volume / sum(
        sum(sp.values()) for sp in [site.species for site in struct])
    if not struct.is_ordered:
        # disordered fcc: occupancies sum to the number of sites; a is the cubic parameter
        a = struct.lattice.abc[0]
        return dict(a=a, c=a, alpha=90.0, a_mean=a)
    a_mean = (4 * vol_per_atom) ** (1 / 3)
    if proto in ("A1", "L1_2"):
        a = prim.lattice.abc[0]
        return dict(a=a, c=a, alpha=90.0, a_mean=a_mean)
    if proto == "L1_0":
        la, lb, lc = prim.lattice.abc
        if len(prim) == 2:  # primitive tetragonal P4/mmm cell: a_t = a / sqrt(2)
            a = la * np.sqrt(2)
        else:
            a = la
        return dict(a=a, c=lc, alpha=90.0, a_mean=a_mean)
    if proto == "L1_1":
        # nearest-neighbour geometry: in-plane NN distance and (111) interplanar spacing along the hex c axis
        conv = SpacegroupAnalyzer(struct, symprec=0.05).get_conventional_standard_structure()
        ah, _, ch = conv.lattice.abc  # hexagonal setting: a_h = in-plane NN, c_h spans 6 (111) layers
        d111 = ch / 6.0
        # face-centred rhombohedral cell with edge a and angle alpha that reproduces (a_h, d111):
        # in-plane NN = a * sqrt((1 - cos(alpha))) ... solved via the deformation in particles.rhombo_strain
        s_perp = ah / (a_mean / np.sqrt(2))
        s_par = d111 / (a_mean / np.sqrt(3))
        # edge length and angle of the deformed cube (see particles.rhombo_strain for the inverse relation)
        a = a_mean * np.sqrt((2 * s_perp ** 2 + s_par ** 2) / 3)
        cos_alpha = (s_par ** 2 - s_perp ** 2) / (3 * (a / a_mean) ** 2)
        return dict(a=a, c=a, alpha=float(np.degrees(np.arccos(cos_alpha))), a_mean=a_mean,
                    s_perp=s_perp, s_par=s_par)
    raise ValueError(proto)


# ----------------------------------------------------------------------------------------------------- COD
COD_URL = "https://www.crystallography.net/cod/result"


def cod_search(elements):
    params = {"format": "json", "strictmin": len(elements), "strictmax": len(elements)}
    for i, el in enumerate(elements, 1):
        params[f"el{i}"] = el
    r = requests.get(COD_URL, params=params, timeout=120)
    r.raise_for_status()
    return r.json()


def cod_fetch_cif(cod_id):
    f = RAW_DIR / "cod" / f"{cod_id}.cif"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        last = None
        for attempt in range(5):  # the COD server drops some TLS connections; retry with backoff
            try:
                r = requests.get(f"https://www.crystallography.net/cod/{cod_id}.cif", timeout=120)
                r.raise_for_status()
                f.write_bytes(r.content)
                time.sleep(0.3)
                return f
            except Exception as e:
                last = e
                time.sleep(2.0 * (attempt + 1))
        raise last
    return f


def parse_cif(path):
    try:
        structs = CifParser(str(path), occupancy_tolerance=1.05).parse_structures(primitive=False)
        return structs[0] if structs else None
    except Exception as e:
        LOG.warning("CIF parse failed %s: %s", path.name, e)
        return None


def cod_candidates():
    out = []
    searches = [["Pt"]] + [[m, "Pt"] for m in ALLOY_ELEMENTS] + [[m] for m in ALLOY_ELEMENTS]
    for els in searches:
        try:
            hits = cod_search(els)
        except Exception as e:
            LOG.warning("COD search %s failed: %s", els, e)
            continue
        LOG.info("COD %s: %d hits", "-".join(els), len(hits))
        for h in hits:
            sg = (h.get("sg") or "").replace(" ", "")
            # keep cubic / tetragonal / rhombohedral close-packed candidates only
            if sg not in ("Fm-3m", "Pm-3m", "P4/mmm", "R-3m", "R-3mH", "I4/mmm", "Fm-3mS"):
                continue
            cid = str(h["file"])
            try:
                f = cod_fetch_cif(cid)
            except Exception as e:
                LOG.warning("COD CIF %s download failed: %s", cid, e)
                continue
            s = parse_cif(f)
            if cid in COD_L11_REBUILD:
                s = l11_from_rhombo(float(h["a"]), float(h["alpha"]), *COD_L11_REBUILD[cid])
            if s is None:
                continue
            pressure = h.get("cellpressure") or h.get("diffrpressure")
            temp = h.get("celltemp") or h.get("diffrtemp") or title_temperature(h.get("title"))
            out.append(dict(source="COD", source_id=cid, structure=s, sg=sg, doi=h.get("doi"),
                            title=(h.get("title") or "")[:120],
                            pressure_kPa=float(pressure) if pressure else None,
                            temp_K=float(temp) if temp else None, method="experimental",
                            year=int(h["year"]) if h.get("year") else None,
                            raw_sha256=sha256_bytes(f.read_bytes())))
    return out


# -------------------------------------------------------------------------------------------------- OQMD
OQMD_URL = "https://oqmd.org/oqmdapi/formationenergy"


def oqmd_candidates():
    out = []
    comps = ["Pt"] + [f"{m}-Pt" for m in ALLOY_ELEMENTS] + ALLOY_ELEMENTS
    cache = RAW_DIR / "oqmd"
    cache.mkdir(parents=True, exist_ok=True)
    for comp in comps:
        f = cache / f"{comp}.json"
        if not f.exists():
            data = []
            offset = 0
            try:
                while True:
                    r = requests.get(OQMD_URL, params={
                        "composition": comp, "limit": 100, "offset": offset,
                        "fields": "name,entry_id,icsd_id,prototype,spacegroup,ntypes,natoms,unit_cell,sites,delta_e,stability"},
                        timeout=240)
                    r.raise_for_status()
                    d = r.json()
                    data += d.get("data", [])
                    if not d.get("meta", {}).get("more_data_available"):
                        break
                    offset += 100
            except Exception as e:
                LOG.warning("OQMD %s failed: %s", comp, e)
                continue
            f.write_text(json.dumps(data))
        data = json.loads(f.read_text())
        LOG.info("OQMD %s: %d entries", comp, len(data))
        for e in data:
            try:
                species, coords = [], []
                for s in e["sites"]:
                    el, xyz = s.split("@")
                    species.append(el.strip())
                    coords.append([float(v) for v in xyz.split()])
                st = Structure(Lattice(e["unit_cell"]), species, coords)
            except Exception:
                continue
            out.append(dict(source="OQMD", source_id=str(e["entry_id"]), structure=st, sg=e.get("spacegroup"),
                            doi="10.1007/s11837-013-0755-4", title=f"OQMD {e.get('prototype')} icsd={e.get('icsd_id')}",
                            year=None, pressure_kPa=None, temp_K=0.0, method="DFT-PBE",
                            stability_eV=e.get("stability"), raw_sha256=sha256_bytes(json.dumps(e, sort_keys=True).encode())))
    return out


# ------------------------------------------------------------------------------------------------ JARVIS
JARVIS_URL = "https://ndownloader.figshare.com/files/64391379"  # jdft_3d-9-24-2025.json.zip (jarvis-tools 2026.6.12)


def jarvis_candidates():
    out = []
    f = RAW_DIR / "jarvis" / "jdft_3d-9-24-2025.json.zip"
    f.parent.mkdir(parents=True, exist_ok=True)
    if not f.exists():
        try:
            r = requests.get(JARVIS_URL, timeout=600)
            r.raise_for_status()
            f.write_bytes(r.content)
        except Exception as e:
            LOG.warning("JARVIS download failed: %s", e)
            return out
    with zipfile.ZipFile(f) as z:
        name = [n for n in z.namelist() if n.endswith(".json")][0]
        data = json.loads(z.read(name))
    allowed = {"Pt"} | set(ALLOY_ELEMENTS)
    for e in data:
        els = set(e["atoms"]["elements"])
        if not els <= allowed or len(els) > 2:
            continue
        if len(els) == 2 and "Pt" not in els:
            continue
        a = e["atoms"]
        st = Structure(Lattice(a["lattice_mat"]), a["elements"], a["coords"], coords_are_cartesian=a.get("cartesian", False))
        out.append(dict(source="JARVIS", source_id=e["jid"], structure=st, sg=e.get("spg_symbol"),
                        doi="10.1038/s41524-020-00440-1", title=f"JARVIS-DFT {e.get('formula')}", year=None,
                        pressure_kPa=None, temp_K=0.0, method="DFT-OptB88vdW",
                        stability_eV=e.get("ehull"), raw_sha256=sha256_bytes(json.dumps(e, sort_keys=True, default=str).encode())))
    LOG.info("JARVIS: %d Pt/M candidates", len(out))
    return out


def mp_candidates():
    if not load_env().get("MP_API_KEY"):
        LOG.info("Materials Project skipped: MP_API_KEY not set in .env")
        return []
    LOG.info("MP_API_KEY present but mp-api is not installed in the pinned environment; skipped")
    return []


# ---------------------------------------------------------------------------------------------- pipeline
def formula_key(struct):
    return struct.composition.reduced_formula


def build_reference_phases():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(STRUCT_DIR / "structures.log", mode="w"), logging.StreamHandler()])
    CIF_DIR.mkdir(parents=True, exist_ok=True)
    cands = cod_candidates() + mp_candidates() + jarvis_candidates() + oqmd_candidates()
    rows = []
    for c in cands:
        st = c["structure"]
        proto, sgn = classify(st)
        c["proto"] = proto
        c["sg_number"] = sgn
        c["formula"] = formula_key(st)
        if proto is None:
            continue
        try:
            c["pc"] = pseudo_cubic_params(st, proto)
        except Exception as e:
            LOG.warning("pseudo-cubic params failed for %s %s: %s", c["source"], c["source_id"], e)
            c["proto"] = None
            continue
        c["kx_converted"] = bool(c["source"] == "COD" and c.get("year") and c["year"] < KX_YEAR)
        if c["kx_converted"]:
            for q in ("a", "c", "a_mean"):
                c["pc"][q] *= KX_TO_A
        rows.append(c)
    LOG.info("classified candidates: %d of %d", len(rows), len(cands))

    # duplicate groups: same reduced formula + prototype + StructureMatcher match (volume-scaled)
    sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True, scale=True, attempt_supercell=True)
    groups = {}
    for c in rows:
        key = (c["formula"], c["proto"])
        placed = False
        for g in groups.setdefault(key, []):
            if c["structure"].is_ordered == g[0]["structure"].is_ordered and (
                    not c["structure"].is_ordered or sm.fit(c["structure"], g[0]["structure"])):
                g.append(c)
                placed = True
                break
            if not c["structure"].is_ordered and not g[0]["structure"].is_ordered:
                g.append(c)
                placed = True
                break
        if not placed:
            groups[key].append([c])

    # ambient experimental Pt lattice parameter, and per-database DFT calibration from that database's Pt
    def ambient(c):
        return c["method"] == "experimental" and (c["pressure_kPa"] is None or c["pressure_kPa"] < 1000) and (
            c["temp_K"] is None or 250 <= c["temp_K"] <= 320) and not EXCLUDE_TITLE.search(c.get("title") or "")

    def robust(members):
        """Drop entries further than 3 scaled MADs (and > 0.005 A) from the median lattice parameter."""
        a = np.array([m["pc"]["a_mean"] for m in members])
        med = np.median(a)
        mad = 1.4826 * np.median(np.abs(a - med))
        keep = np.abs(a - med) <= max(3 * mad, 0.005)
        return [m for m, k in zip(members, keep) if k]

    def rep_value(members, key):
        return float(np.median([m["pc"][key] for m in members]))

    ref = {}
    pt_exp = robust([c for c in rows if c["formula"] == "Pt" and c["proto"] == "A1" and ambient(c)])
    a_pt_exp = float(np.median([c["pc"]["a"] for c in pt_exp]))
    dft_scale = {}
    for src in ("JARVIS", "OQMD", "MP"):
        pts = [c for c in rows if c["source"] == src and c["formula"] == "Pt" and c["proto"] == "A1"]
        if pts:
            a_dft = min(pts, key=lambda c: c.get("stability_eV") or 0.0)["pc"]["a"]
            dft_scale[src] = a_pt_exp / a_dft
    LOG.info("a(Pt) exp median = %.4f A from %d COD entries; DFT scale factors %s", a_pt_exp, len(pt_exp), dft_scale)

    phase_targets = [("Pt", "A1")] + [(m, "A1") for m in ALLOY_ELEMENTS] + [
        (sys_name, spec["ordered"]) for sys_name, spec in SYSTEMS.items() if spec["ordered"]]
    # also record database values for disordered alloys at any composition (Vegard check)
    for (formula, proto) in phase_targets:
        key_formula = {"PtCo": "CoPt", "PtNi": "NiPt", "PtCu": "CuPt", "Pt3Co": "CoPt3", "Pt3Ni": "NiPt3",
                       "Pt3Fe": "FePt3"}.get(formula, formula)
        members = [c for g in groups.get((key_formula, proto), []) for c in g]
        if not members:
            LOG.warning("No open structure for %s %s - phase skipped", formula, proto)
            ref[f"{formula}|{proto}"] = dict(formula=formula, proto=proto, available=False)
            continue
        exp = robust([m for m in members if ambient(m)]) if any(ambient(m) for m in members) else []
        gamma_fe = [m for m in members if formula == "Fe" and m["method"] == "experimental" and m["temp_K"]
                    and m["temp_K"] > 1000 and "lattice expansion" in (m.get("title") or "").lower()]
        if formula == "Fe" and len(gamma_fe) >= 3:
            # fcc Fe is not stable at room temperature: extrapolate the high-temperature gamma-Fe series to 293 K
            T = np.array([m["temp_K"] for m in gamma_fe])
            A = np.array([m["pc"]["a"] for m in gamma_fe])
            slope, icpt = np.polyfit(T, A, 1)
            a293 = float(icpt + slope * 293.15)
            rep = gamma_fe[0]
            ref["Fe|A1"] = dict(
                formula="Fe", proto="A1", available=True,
                basis=f"gamma-Fe a(T) series ({len(gamma_fe)} COD entries, {T.min():.0f}-{T.max():.0f} K) linearly "
                      f"extrapolated to 293 K (slope {slope:.2e} A/K)",
                params=dict(a=a293, c=a293, alpha=90.0, a_mean=a293),
                representative=dict(source="COD", source_id=",".join(m["source_id"] for m in gamma_fe),
                                    licence=LICENCES["COD"], doi=rep.get("doi"), cif=None),
                members=[dict(source=m["source"], source_id=m["source_id"], a=m["pc"]["a"], c=m["pc"]["c"],
                              method=m["method"], pressure_kPa=m["pressure_kPa"], temp_K=m["temp_K"]) for m in members])
            LOG.info("Fe     A1    a=%.4f  <- %s", a293, ref["Fe|A1"]["basis"])
            continue
        if exp:
            chosen, basis = exp, f"experimental (COD, ambient), median of {len(exp)} entries after outlier rejection"
            scale = 1.0
        else:
            # DFT only: lowest-energy-above-hull entries per source, rescaled with that source's Pt calibration
            by_src = {}
            for m in members:
                if m["source"] in dft_scale:
                    by_src.setdefault(m["source"], []).append(m)
            if not by_src:
                ref[f"{formula}|{proto}"] = dict(formula=formula, proto=proto, available=False)
                LOG.warning("No calibratable structure for %s %s - phase skipped", formula, proto)
                continue
            src = "JARVIS" if "JARVIS" in by_src else sorted(by_src)[0]
            chosen = [min(by_src[src], key=lambda m: m.get("stability_eV") or 0.0)]
            scale = dft_scale[src]
            basis = f"{src} DFT, isotropically rescaled by a(Pt,exp)/a(Pt,{src}) = {scale:.4f}"
        rep = chosen[0]
        pc = {k: rep_value(chosen, k) * (scale if k in ("a", "c", "a_mean") else 1.0)
              for k in rep["pc"] if isinstance(rep["pc"][k], float)}
        if proto == "L1_1":
            pc["alpha"] = rep_value(chosen, "alpha")
        cif_name = f"{rep['source']}_{rep['source_id']}.cif"
        rep["structure"].to(filename=str(CIF_DIR / cif_name))
        ref[f"{formula}|{proto}"] = dict(
            formula=formula, proto=proto, available=True, basis=basis, params=pc,
            representative=dict(source=rep["source"], source_id=rep["source_id"], licence=LICENCES[rep["source"]],
                                doi=rep.get("doi"), cif=f"cif/{cif_name}"),
            members=[dict(source=m["source"], source_id=m["source_id"], a=m["pc"]["a"], c=m["pc"]["c"],
                          method=m["method"], pressure_kPa=m["pressure_kPa"], temp_K=m["temp_K"],
                          kx_converted=m.get("kx_converted", False), used=any(m is x for x in chosen))
                     for m in members])
        LOG.info("%-6s %-5s a=%.4f c=%.4f alpha=%.2f  <- %s (%d duplicates)", formula, proto, pc.get("a"), pc.get("c"),
                 pc.get("alpha", 90.0), basis, len(members))

    # Vegard's law for disordered A1 alloys, checked against database A1 entries with mixed occupancy
    vegard = {}
    a_end = {el: ref[f"{el}|A1"]["params"]["a"] for el in ["Pt"] + ALLOY_ELEMENTS if ref.get(f"{el}|A1", {}).get("available")}
    if "Fe" in a_end:
        # COD fcc-Fe entries are high-temperature gamma-Fe; record what was used
        pass
    for sys_name, spec in SYSTEMS.items():
        if spec["M"] is None:
            continue
        m = spec["M"]
        x = spec["x"]
        a_v = (1 - x) * a_end["Pt"] + x * a_end[m]
        checks = []
        for c in rows:
            if c["proto"] == "A1" and not c["structure"].is_ordered:
                comp = c["structure"].composition.fractional_composition
                if set(e.symbol for e in comp.elements) == {"Pt", m}:
                    xm = comp[m]
                    a_vx = (1 - xm) * a_end["Pt"] + xm * a_end[m]
                    checks.append(dict(source=c["source"], source_id=c["source_id"], x_M=round(float(xm), 3),
                                       a_db=round(c["pc"]["a"], 4), a_vegard=round(a_vx, 4),
                                       rel_dev=round((c["pc"]["a"] - a_vx) / a_vx, 4)))
        for key in (f"{sys_name}|{spec['ordered']}",):
            if ref.get(key, {}).get("available"):
                am = ref[key]["params"]["a_mean"]
                checks.append(dict(source="ordered phase (volume-equivalent)", source_id=ref[key]["representative"]["source_id"],
                                   x_M=x, a_db=round(am, 4), a_vegard=round(a_v, 4), rel_dev=round((am - a_v) / a_v, 4)))
        vegard[sys_name] = dict(M=m, x_M=x, a_vegard=a_v, checks=checks)
        LOG.info("Vegard %-6s a=%.4f ; database checks: %s", sys_name, a_v,
                 "; ".join(f"{c['source']}:{c['source_id']} x={c['x_M']} a={c['a_db']} dev={100*c['rel_dev']:+.2f}%" for c in checks) or "none")

    out = dict(a_pt_exp=a_pt_exp, dft_scale=dft_scale, endpoints=a_end, phases=ref, vegard=vegard)
    (STRUCT_DIR / "reference_phases.json").write_text(json.dumps(out, indent=2))

    with open(STRUCT_DIR / "candidates.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "source_id", "licence", "formula", "prototype", "space_group", "method", "a", "c",
                    "pressure_kPa", "temp_K", "doi", "title", "sha256"])
        for c in cands:
            pc = c.get("pc") or {}
            w.writerow([c["source"], c["source_id"], LICENCES[c["source"]], c.get("formula"), c.get("proto"),
                        c.get("sg"), c["method"], pc.get("a"), pc.get("c"), c["pressure_kPa"], c["temp_K"], c.get("doi"),
                        c.get("title"), c["raw_sha256"]])
    return out


if __name__ == "__main__":
    build_reference_phases()
