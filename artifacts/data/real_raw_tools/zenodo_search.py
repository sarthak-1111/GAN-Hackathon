"""Search Zenodo records via the public REST API and print a compact summary.

Usage: python -I zenodo_search.py "<query>" [n_results] [--files] [--desc]
(unauthenticated page size is limited to 25; more results are fetched page by page)
"""
import re
import sys
import time

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    q = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 25
    show_files = "--files" in sys.argv
    show_desc = "--desc" in sys.argv
    got, page = 0, 1
    while got < n:
        r = requests.get("https://zenodo.org/api/records",
                         params={"q": q, "size": min(25, n - got), "page": page, "sort": "bestmatch"}, timeout=90)
        if r.status_code != 200:
            print("HTTP", r.status_code, r.text[:300])
            return
        d = r.json()
        if page == 1:
            print("total", d["hits"]["total"])
        hits = d["hits"]["hits"]
        if not hits:
            break
        for h in hits:
            m = h["metadata"]
            files = h.get("files", [])
            tot = sum(f["size"] for f in files)
            rt = m.get("resource_type", {}).get("type")
            print(f"{h['id']} | {m.get('publication_date')} | {m.get('license', {}).get('id')} | {rt} | "
                  f"{len(files)} files {tot/1e6:.1f} MB | {m['title'][:150]}")
            if show_files:
                for f in files[:40]:
                    print(f"      {f['key']}  {f['size']}")
            if show_desc:
                desc = re.sub(r"<[^>]+>", " ", m.get("description", ""))
                print("      DESC:", re.sub(r"\s+", " ", desc)[:600])
        got += len(hits)
        page += 1
        time.sleep(1)


if __name__ == "__main__":
    main()
