"""Scan Zenodo search results (many pages) and report records with data-like files.

Usage: python -I zenodo_scan.py "<query>" <max_records> <out_json>
Writes all hits (id, title, licence, type, files) to out_json and prints records whose
file list contains spectra-like extensions or archives.
"""
import json
import re
import sys
import time

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DATA_EXT = re.compile(r"\.(gr|fgr|sq|fq|iq|chi|xy|xye|xrdml|brml|raw|dat|csv|txt|xdi|xmu|nor|prj|h5|hdf5|nxs|zip|tar|gz|7z|npy)$", re.I)


def main():
    q, nmax, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    allh = []
    page = 1
    while len(allh) < nmax:
        for attempt in range(3):
            r = requests.get("https://zenodo.org/api/records",
                             params={"q": q, "size": 25, "page": page, "sort": "bestmatch"}, timeout=120)
            if r.status_code == 200:
                break
            time.sleep(5)
        if r.status_code != 200:
            print("HTTP", r.status_code, r.text[:200])
            break
        hits = r.json()["hits"]["hits"]
        if not hits:
            break
        for h in hits:
            m = h["metadata"]
            rec = dict(id=h["id"], title=m["title"], date=m.get("publication_date"),
                       licence=(m.get("license") or {}).get("id"), type=m.get("resource_type", {}).get("type"),
                       creators=[c["name"] for c in m.get("creators", [])][:6],
                       files=[(f["key"], f["size"]) for f in h.get("files", [])])
            allh.append(rec)
        page += 1
        time.sleep(1.2)
    json.dump(allh, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    for rec in allh:
        dfiles = [f for f in rec["files"] if DATA_EXT.search(f[0])]
        if rec["type"] in ("dataset", "other", "software") and dfiles:
            tot = sum(s for _, s in rec["files"]) / 1e6
            ex = ", ".join(f"{k}" for k, _ in dfiles[:6])
            print(f"{rec['id']} | {rec['date']} | {rec['licence']} | {rec['type']} | {tot:.1f} MB | {rec['title'][:110]} || {ex}")
    print("scanned", len(allh))


if __name__ == "__main__":
    main()
