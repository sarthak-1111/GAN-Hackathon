"""Download selected files of a Zenodo record (by regex on file key) into a directory.

Also saves the record metadata JSON (_zenodo_record.json) for provenance/licence.
Usage: python -I zenodo_download.py <record_id> <out_dir> [key_regex] [--max-mb N]
"""
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    rid, out = sys.argv[1], Path(sys.argv[2])
    rx = re.compile(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else None
    max_mb = float(sys.argv[sys.argv.index("--max-mb") + 1]) if "--max-mb" in sys.argv else 1e9
    out.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = "GAN-realism-dataset-collector/0.1 (research)"
    rec = s.get(f"https://zenodo.org/api/records/{rid}", timeout=120).json()
    (out / "_zenodo_record.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    m = rec["metadata"]
    print(rid, m["title"][:120], "| licence:", (m.get("license") or {}).get("id"))
    for f in rec.get("files", []):
        key = f["key"]
        if rx and not rx.search(key):
            continue
        if f["size"] > max_mb * 1e6:
            print("skip (too big)", key, f["size"])
            continue
        safe = re.sub(r"[^A-Za-z0-9._+()%-]", "_", key.replace("/", "__")).replace("%", "pct")
        dest = out / safe
        if dest.exists() and dest.stat().st_size == f["size"]:
            continue
        url = f"https://zenodo.org/api/records/{rid}/files/{quote(key, safe='')}/content"
        for attempt in range(4):
            try:
                with s.get(url, stream=True, timeout=300) as r:
                    r.raise_for_status()
                    with open(dest, "wb") as fh:
                        for chunk in r.iter_content(1 << 20):
                            fh.write(chunk)
                if dest.stat().st_size == f["size"]:
                    break
            except requests.RequestException as e:
                print("retry", key, e)
            time.sleep(5 * (attempt + 1))
        print("saved", safe, dest.stat().st_size, "md5:", f.get("checksum"))
        time.sleep(0.5)


if __name__ == "__main__":
    main()
