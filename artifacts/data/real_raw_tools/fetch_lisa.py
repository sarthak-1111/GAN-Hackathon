"""Download XDI files from the LISA (ESRF BM08) XAS Database web site.

Site: https://lisa.iom.cnr.it/xasdb/   (footer: "Copyright Public Domain";
cite via Zenodo concept DOI 10.5281/zenodo.10778068, record licence CC-BY-4.0)

Usage: python -I fetch_lisa.py <out_dir>
"""
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = "https://lisa.iom.cnr.it/xasdb/"


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = "GAN-realism-dataset-collector/0.1 (research)"
    idx = s.get(BASE, timeout=60).text
    pages = sorted(set(re.findall(r'href="\./([A-Z][a-z]?)\.html"', idx)))
    index = []
    for el in pages:
        html = s.get(urljoin(BASE, f"{el}.html"), timeout=60).text
        links = re.findall(r'href="(\./data/[^"]+)"', html)
        # label text that follows each link
        for link in links:
            url = urljoin(BASE, link)
            fname = link.rsplit("/", 1)[-1]
            dest = out / el / fname
            dest.parent.mkdir(parents=True, exist_ok=True)
            rec = dict(element=el, url=url, file=f"{el}/{fname}", page=urljoin(BASE, f"{el}.html"))
            index.append(rec)
            if dest.exists() and dest.stat().st_size > 0:
                continue
            r = s.get(url, timeout=60)
            if r.status_code != 200:
                rec["error"] = f"HTTP {r.status_code}"
                print("FAIL", url, r.status_code, flush=True)
                continue
            dest.write_bytes(r.content)
            print("saved", el, fname, len(r.content), flush=True)
            time.sleep(0.3)
        time.sleep(0.3)
    (out / "_index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    print("done", len(index), flush=True)


if __name__ == "__main__":
    main()
