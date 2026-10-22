"""Build an index of MDR XAFS DB datasets from DataCite metadata (fast, public API).

Usage: python -I mdr_datacite_index.py <out_json>
"""
import json
import sys
import time

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    out = sys.argv[1]
    recs = []
    url = "https://api.datacite.org/dois"
    params = {"query": 'titles.title:"XAFS spectrum"', "prefix": "10.48505",
              "page[size]": 1000, "page[cursor]": 1}
    while url:
        r = requests.get(url, params=params, timeout=120)
        r.raise_for_status()
        d = r.json()
        for x in d["data"]:
            a = x["attributes"]
            recs.append(dict(
                doi=x["id"],
                title=a["titles"][0]["title"] if a.get("titles") else "",
                url=a.get("url"),
                rights=[(rr.get("rightsIdentifier") or rr.get("rights") or "") for rr in a.get("rightsList", [])],
                rights_uri=[rr.get("rightsUri", "") for rr in a.get("rightsList", [])],
                subjects=[s.get("subject") for s in a.get("subjects", [])],
                descriptions=[dd.get("description") for dd in a.get("descriptions", [])][:1],
                creators=[c.get("name") for c in a.get("creators", [])],
                publisher=a.get("publisher"),
                year=a.get("publicationYear"),
                updated=a.get("updated"),
            ))
        nxt = d.get("links", {}).get("next")
        print(len(recs), "next:", bool(nxt), flush=True)
        url, params = (nxt, None) if nxt else (None, None)
        time.sleep(1)
    json.dump(recs, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print("saved", len(recs))


if __name__ == "__main__":
    main()
