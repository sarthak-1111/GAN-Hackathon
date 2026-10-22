"""Safely extract (a regex-selected subset of) a local ZIP archive into a NEW directory.

- rejects absolute paths / '..' components (no path traversal)
- never executes anything; only writes bytes
Usage: python -I safe_unzip.py <zip> <out_dir> [include_regex] [exclude_regex]
"""
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    zpath, out = sys.argv[1], Path(sys.argv[2])
    inc = re.compile(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None
    exc = re.compile(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else None
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    tot = 0
    with zipfile.ZipFile(zpath) as z:
        for m in z.infolist():
            if m.is_dir():
                continue
            name = m.filename.replace("\\", "/")
            if inc and not inc.search(name):
                continue
            if exc and exc.search(name):
                continue
            p = PurePosixPath(name)
            if p.is_absolute() or ".." in p.parts or ":" in name:
                print("SKIP unsafe", name)
                continue
            dest = out.joinpath(*p.parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, open(dest, "wb") as dst:
                while True:
                    b = src.read(1 << 20)
                    if not b:
                        break
                    dst.write(b)
            n += 1
            tot += m.file_size
    print("extracted", n, "files", tot, "bytes")


if __name__ == "__main__":
    main()
