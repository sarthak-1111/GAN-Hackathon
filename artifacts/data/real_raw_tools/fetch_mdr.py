"""Download MDR XAFS DB dataset ZIPs (NIMS Materials Data Repository).

Selection: DataCite index (mdr_datacite_index.py output) filtered by edge subjects and
licence == cc-by-4.0. Each dataset ZIP: https://mdr.nims.go.jp/datasets/<uuid>.zip
(documented in the MDR user's manual, API -> Individual works -> Download).

Usage: python -I fetch_mdr.py <index_json> <out_dir> [n_workers]
"""
import json
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
WANT = {"Pt L3-edge", "Co K-edge", "Ni K-edge", "Fe K-edge", "Cu K-edge"}
lock = threading.Lock()


def fetch(rec, out):
    uuid = rec["url"].rstrip("/").rsplit("/", 1)[-1]
    tag = rec["doi"].split("/")[-1].replace("nims.", "nims_")
    zpath = out / "zips" / f"{tag}_{uuid}.zip"
    xdir = out / "extracted" / tag
    rec["uuid"], rec["zip"], rec["dir"] = uuid, str(zpath.relative_to(out)), str(xdir.relative_to(out))
    if not (zpath.exists() and zipfile.is_zipfile(zpath)):
        ok = False
        for attempt in range(5):
            try:
                r = requests.get(f"https://mdr.nims.go.jp/datasets/{uuid}.zip", timeout=180,
                                 headers={"User-Agent": "GAN-realism-dataset-collector/0.1 (research)"})
                if r.status_code == 200 and r.content[:2] == b"PK":
                    zpath.write_bytes(r.content)
                    ok = True
                    break
                with lock:
                    print("HTTP", r.status_code, tag, "attempt", attempt, flush=True)
            except requests.RequestException as e:
                with lock:
                    print("ERR", tag, e, flush=True)
            time.sleep(10 * (attempt + 1))
        if not ok:
            rec["error"] = "download failed"
            return rec
    xdir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        for m in z.infolist():
            name = Path(m.filename).name  # flatten, avoid path traversal
            if not name or m.is_dir():
                continue
            (xdir / name).write_bytes(z.read(m))
        rec["files"] = sorted(Path(m.filename).name for m in z.infolist() if not m.is_dir())
    with lock:
        print("ok", tag, len(rec["files"]), flush=True)
    time.sleep(1.0)
    return rec


def main():
    idx = json.load(open(sys.argv[1], encoding="utf-8"))
    out = Path(sys.argv[2])
    nw = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    (out / "zips").mkdir(parents=True, exist_ok=True)
    (out / "extracted").mkdir(parents=True, exist_ok=True)
    sel = [x for x in idx if WANT & set(x["subjects"]) and x["rights"] == ["cc-by-4.0"]]
    if "--reverse" in sys.argv:  # second helper process working from the other end
        sel = sel[::-1]
    print("selected", len(sel), flush=True)
    with ThreadPoolExecutor(nw) as ex:
        res = list(ex.map(lambda r: fetch(r, out), sel))
    json.dump(res, open(out / "_downloaded.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("done", sum("error" not in r for r in res), "of", len(res), flush=True)


if __name__ == "__main__":
    main()
