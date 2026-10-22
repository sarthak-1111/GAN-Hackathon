"""Download spectra from the Canadian Light Source XAS Database (XASDB).

Licence: "All data in the XASDB is provided under the Creative Commons Attribution 4.0
International (CC BY 4.0) license" (https://xasdb.lightsource.ca/views/about.html);
each spectrum header also carries Scan.license.

API (documented in the About -> Search & API tab):
  POST /search          {"element": "Fe"}   -> {"cmds": [{compound, sids, spectra:[{sid, beamline, edge, compound}]}], ...}
  POST /xasgetcompounds {"id": sid, "element": El, "name": compound} -> {"spectrum": {...full data...}}

Usage: python -I fetch_cls_xasdb.py <out_dir> [El1,El2,...]
Saves raw JSON (data only) to <out_dir>/json/<El>/<sid>.json and an index to <out_dir>/_index.json
"""
import json
import sys
import time
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = "https://xasdb.lightsource.ca"
ELEMENTS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
            "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb "
            "Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm").split()


def post(sess, path, payload):
    for attempt in range(4):
        try:
            r = sess.post(BASE + path, json=payload, timeout=90)
            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError:
                    return None
            print("HTTP", r.status_code, path, payload, flush=True)
        except requests.RequestException as e:
            print("ERR", e, path, payload, flush=True)
        time.sleep(5 * (attempt + 1))
    return None


def main():
    out = Path(sys.argv[1])
    els = sys.argv[2].split(",") if len(sys.argv) > 2 else ELEMENTS
    (out / "json").mkdir(parents=True, exist_ok=True)
    idx_path = out / "_index.json"
    index = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    sess = requests.Session()
    sess.headers.update({"User-Agent": "GAN-realism-dataset-collector/0.1 (research)",
                         "Content-Type": "application/json; charset=utf-8"})
    for el in els:
        res = post(sess, "/search", {"element": el})
        time.sleep(0.5)
        if not res or not isinstance(res, dict) or not res.get("cmds"):
            continue
        n = 0
        for cmd in res["cmds"]:
            for sp in cmd.get("spectra", []):
                sid = sp.get("sid")
                if not sid:
                    continue
                n += 1
                el_of_sid = sid.split("_")[0]
                dest = out / "json" / el_of_sid / f"{sid}.json"
                rec = dict(sid=sid, element=el_of_sid, compound=cmd.get("compound"),
                           beamline=sp.get("beamline"), edge=sp.get("edge"),
                           file=str(dest.relative_to(out)).replace("\\", "/"))
                index[sid] = rec
                if dest.exists() and dest.stat().st_size > 100:
                    continue
                data = post(sess, "/xasgetcompounds", {"id": sid, "element": el_of_sid, "name": cmd.get("compound")})
                time.sleep(0.4)
                if not data or not isinstance(data, dict) or not data.get("spectrum"):
                    rec["error"] = "fetch failed"
                    print("FAIL", sid, flush=True)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps(data["spectrum"]), encoding="utf-8")
        print(el, "spectra:", n, flush=True)
        idx_path.write_text(json.dumps(index, indent=1), encoding="utf-8")
    idx_path.write_text(json.dumps(index, indent=1), encoding="utf-8")
    print("done", len(index), flush=True)


if __name__ == "__main__":
    main()
