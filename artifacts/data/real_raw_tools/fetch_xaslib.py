"""Download all spectra from the IXAS X-ray Absorption Data Library (XASLIB).

Licence (stated at https://docs.xrayabsorption.org/xaslib/license.html):
CC0 / Public Domain.

Usage:  python -I fetch_xaslib.py <out_dir>
Writes <out_dir>/<Element>/<id>_<file>.xdi and <out_dir>/_index.json
Only parses HTML / saves files as data; never executes downloaded content.
"""
import json
import re
import sys
import time
from pathlib import Path

import requests

BASE = "https://xaslib.xrayabsorption.org"
UA = {"User-Agent": "GAN-realism-dataset-collector/0.1 (research; contact via project owner)"}


def get(sess, url, **kw):
    for attempt in range(4):
        try:
            r = sess.get(url, timeout=60, **kw)
            if r.status_code == 200:
                return r
            print("HTTP", r.status_code, url, flush=True)
        except requests.RequestException as e:
            print("ERR", e, url, flush=True)
        time.sleep(3 * (attempt + 1))
    return None


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    sess.headers.update(UA)
    home = get(sess, BASE + "/").text
    elems = sorted(set(re.findall(r'href="/elem/([A-Z][a-z]?)"', home)))
    print("elements:", elems, flush=True)
    index = []
    row_re = re.compile(
        r'<a href="/spectrum/(\d+)/">\s*([^<]*)</a></td>\s*<td>\s*(?:&nbsp;)?\s*([^<]*)</td>\s*<td>\s*(?:&nbsp;)?\s*([^<]*)</td>',
        re.S)
    for el in elems:
        page = get(sess, f"{BASE}/elem/{el}")
        if page is None:
            continue
        rows = row_re.findall(page.text)
        print(el, len(rows), flush=True)
        for sid, name, edge, beamline in rows:
            index.append(dict(id=int(sid), name=name.strip(), edge=edge.strip(),
                              beamline=beamline.strip(), element=el))
        time.sleep(0.5)
    # fetch each spectrum page for raw file link, then the raw file
    for rec in index:
        sid = rec["id"]
        page = get(sess, f"{BASE}/spectrum/{sid}/")
        if page is None:
            rec["error"] = "spectrum page failed"
            continue
        m = re.search(r'href="(/rawfile/%d/[^"]+)"' % sid, page.text)
        if not m:
            rec["error"] = "no rawfile link"
            continue
        link = m.group(1)
        fname = link.rsplit("/", 1)[-1]
        safe = re.sub(r"[^A-Za-z0-9._+-]", "_", fname)
        d = out / rec["element"]
        d.mkdir(exist_ok=True)
        dest = d / f"{sid:04d}_{safe}"
        rec["url"] = BASE + link
        rec["file"] = str(dest.relative_to(out)).replace("\\", "/")
        if dest.exists() and dest.stat().st_size > 0:
            continue
        raw = get(sess, BASE + link)
        if raw is None:
            rec["error"] = "rawfile download failed"
            continue
        dest.write_bytes(raw.content)
        print("saved", dest.name, len(raw.content), flush=True)
        time.sleep(0.4)
    (out / "_index.json").write_text(json.dumps(index, indent=1))
    print("done", len(index), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
