"""Extract spectra from Athena project files (.prj) by plain TEXT PARSING (no eval / no Perl / no asteval).

Old-style Athena/Demeter projects are (optionally gzipped) Perl Data::Dumper text with lines like
   $old_group = 'abcd';
   @args = ('label','Pt foil','datatype','xmu',...);
   @x = ('11400.1','11400.6',...);
   @y = ('0.12','0.13',...);
   [record] ...
New-style projects are JSON (handled with json.loads).

Usage: python -I athena_prj_extract.py <in.prj> <out_dir>
Writes <out_dir>/<prj-stem>__<NN>_<label>.dat (columns: x y [i0]) and an index JSON.
"""
import gzip
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
QUOTED = re.compile(r"'((?:[^'\\]|\\.)*)'")


def read_text(p):
    b = Path(p).read_bytes()
    if b[:2] == b"\x1f\x8b":
        b = gzip.decompress(b)
    return b.decode("latin-1")


def plist(line):
    """Return list of tokens inside the outer (...) of a Perl list assignment.

    Tokens are comma-separated; each is either a single-quoted string (with \\' escapes)
    or a bare word / number. Order is preserved (needed for key,value pairing of @args).
    """
    i, j = line.find("("), line.rfind(")")
    if i < 0 or j < 0:
        return []
    body = line[i + 1:j]
    toks, k, n = [], 0, len(body)
    while k < n:
        c = body[k]
        if c in " \t,":
            k += 1
            continue
        if c == "'":
            k += 1
            buf = []
            while k < n:
                if body[k] == "\\" and k + 1 < n:
                    buf.append(body[k + 1])
                    k += 2
                    continue
                if body[k] == "'":
                    k += 1
                    break
                buf.append(body[k])
                k += 1
            toks.append("".join(buf))
        else:
            e = body.find(",", k)
            e = n if e < 0 else e
            toks.append(body[k:e].strip())
            k = e
    return toks


def parse_old(text):
    groups, cur = [], {}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("[record]"):
            if cur:
                groups.append(cur)
            cur = {}
            continue
        m = re.match(r"^[\$@%]?(\w+)\s*=", s)
        if not m:
            continue
        key = m.group(1)
        if key == "old_group":
            q = QUOTED.findall(s)
            cur["old_group"] = q[0] if q else ""
        elif key == "args":
            a = plist(s)
            cur["args"] = {a[k]: a[k + 1] for k in range(0, len(a) - 1, 2)}
        elif key in ("x", "y", "i0", "signal", "stddev"):
            vals = []
            for t in plist(s):
                try:
                    vals.append(float(t))
                except ValueError:
                    vals.append(float("nan"))
            cur[key] = vals
    if cur:
        groups.append(cur)
    return groups


def parse_new(text):
    d = json.loads(text)
    out = []
    for g in d.get("groups", []) if isinstance(d, dict) else []:
        out.append(g)
    return out


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    text = read_text(src)
    first = text.lstrip()[:1]
    groups = parse_new(text) if first == "{" else parse_old(text)
    index = []
    for n, g in enumerate(groups):
        x, y = g.get("x"), g.get("y")
        if not x or not y or len(x) != len(y):
            continue
        args = g.get("args", {})
        label = args.get("label") or g.get("old_group") or f"group{n}"
        safe = re.sub(r"[^A-Za-z0-9._+-]", "_", label)[:60]
        dest = out / f"{src.stem}__{n:02d}_{safe}.dat"
        i0 = g.get("i0") if g.get("i0") and len(g.get("i0")) == len(x) else None
        meta = {"orig_file": args.get("file")} if args.get("file") else {}
        meta.update({k: args.get(k) for k in ("label", "datatype", "is_xmu", "is_chi", "is_nor", "bkg_e0", "bkg_z",
                                          "fft_edge", "bkg_eshift", "importance", "is_merge",
                                          "bkg_nnorm", "is_fixed") if k in args})
        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(f"# extracted from Athena project {src.name} (group {n}) by plain-text parsing\n")
            for k, v in meta.items():
                fh.write(f"# {k}: {v}\n")
            fh.write("# columns: x y" + (" i0" if i0 else "") + "   (x = energy eV for datatype xmu; k for chi)\n")
            for k in range(len(x)):
                fh.write(f"{x[k]:.6f} {y[k]:.8g}" + (f" {i0[k]:.8g}" if i0 else "") + "\n")
        index.append(dict(file=dest.name, group=n, npts=len(x), xmin=min(x), xmax=max(x), **meta))
    (out / f"{src.stem}__index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    print(src.name, "groups:", len(groups), "written:", len(index))
    for r in index:
        print("  ", r["file"], r.get("datatype"), r.get("bkg_e0"), r["npts"], round(r["xmin"], 1), round(r["xmax"], 1))


if __name__ == "__main__":
    main()
