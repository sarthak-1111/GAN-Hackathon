"""Search DataCite (all repositories: Zenodo, Figshare, Dryad, Mendeley Data, institutional...) for datasets.

Usage: python -I datacite_search.py "<elasticsearch query>" [max] [--all-types]
Prints: doi | year | publisher | rights | title | url
"""
import sys
import time

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    q = sys.argv[1]
    nmax = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 100
    params = {"query": q, "page[size]": min(nmax, 1000), "page[cursor]": 1,
              "fields[dois]": "titles,publisher,publicationYear,rightsList,url,types,sizes,formats"}
    if "--all-types" not in sys.argv:
        params["resource-type-id"] = "dataset"
    url = "https://api.datacite.org/dois"
    n = 0
    while url and n < nmax:
        r = requests.get(url, params=params, timeout=120)
        if r.status_code != 200:
            print("HTTP", r.status_code, r.text[:300])
            return
        d = r.json()
        if n == 0:
            print("total", d.get("meta", {}).get("total"))
        for x in d["data"]:
            a = x["attributes"]
            rights = ";".join((rr.get("rightsIdentifier") or rr.get("rights") or "")[:40] for rr in a.get("rightsList", []))
            title = a["titles"][0]["title"] if a.get("titles") else ""
            print(f"{x['id']} | {a.get('publicationYear')} | {str(a.get('publisher'))[:30]} | {rights} | {title[:140]} | {a.get('url')}")
            n += 1
            if n >= nmax:
                break
        url = d.get("links", {}).get("next")
        params = None
        time.sleep(1)


if __name__ == "__main__":
    main()
