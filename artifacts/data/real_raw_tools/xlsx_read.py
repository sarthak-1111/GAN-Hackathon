"""Minimal, dependency-free XLSX reader (zip + XML parsing only; no macros, nothing executed).

Usage as script:  python -I xlsx_read.py <file.xlsx> [--sheets] [--dump <sheet_name> <out_csv>] [--head <sheet_name> N]
Library use: read_sheets(path) -> {sheet_name: [[cell, ...], ...]}
"""
import csv
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}


def col_index(ref):
    letters = re.match(r"([A-Z]+)", ref).group(1)
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def read_sheets(path, only=None):
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in root.findall("m:si", NS):
            shared.append("".join(t.text or "" for t in si.iter("{%s}t" % NS["m"])))
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    rid2target = {r.get("Id"): r.get("Target") for r in rels}
    out = {}
    for sh in wb.find("m:sheets", NS):
        name = sh.get("name")
        if only and name not in only:
            continue
        rid = sh.get("{%s}id" % NS["r"])
        target = rid2target[rid].lstrip("/")
        if not target.startswith("xl/"):
            target = "xl/" + target
        root = ET.fromstring(z.read(target))
        rows = []
        for row in root.iter("{%s}row" % NS["m"]):
            vals = {}
            for c in row.findall("m:c", NS):
                ref, t = c.get("r"), c.get("t")
                v = c.find("m:v", NS)
                if t == "s" and v is not None:
                    val = shared[int(v.text)]
                elif t == "inlineStr":
                    val = "".join(x.text or "" for x in c.iter("{%s}t" % NS["m"]))
                else:
                    val = v.text if v is not None else ""
                vals[col_index(ref)] = val
            if vals:
                mx = max(vals)
                rows.append([vals.get(i, "") for i in range(mx + 1)])
            else:
                rows.append([])
        out[name] = rows
    return out


def main():
    p = sys.argv[1]
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if "--sheets" in sys.argv:
        s = read_sheets(p)
        for k, v in s.items():
            ncol = max((len(r) for r in v), default=0)
            print(f"{k}: {len(v)} rows x {ncol} cols | first row: {v[0][:12] if v else ''}")
    if "--head" in sys.argv:
        i = sys.argv.index("--head")
        name, n = sys.argv[i + 1], int(sys.argv[i + 2])
        rows = read_sheets(p, only={name})[name]
        for r in rows[:n]:
            print(r[:16])
    if "--dump" in sys.argv:
        i = sys.argv.index("--dump")
        name, outp = sys.argv[i + 1], sys.argv[i + 2]
        rows = read_sheets(p, only={name})[name]
        with open(outp, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(rows)


if __name__ == "__main__":
    main()
