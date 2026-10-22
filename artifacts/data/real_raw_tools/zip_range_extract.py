"""List / selectively extract members of a remote ZIP using HTTP Range requests.

Avoids downloading multi-GB archives when only a few data files are needed.
Data are only read and written; nothing from the archive is executed.

Usage:
  python -I zip_range_extract.py list    <url> [regex]
  python -I zip_range_extract.py extract <url> <regex> <out_dir>
Members are written flattened (basename only, prefixed by a sanitised parent-folder tag)
so no path traversal is possible.
"""
import io
import re
import sys
import zipfile
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BLOCK = 1 << 20  # 1 MiB


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "GAN-realism-dataset-collector/0.1 (research)"
        r = self.s.get(url, headers={"Range": "bytes=0-0"}, timeout=120, allow_redirects=True)
        if r.status_code != 206:
            raise RuntimeError(f"range requests not supported: HTTP {r.status_code}")
        self.size = int(r.headers["Content-Range"].split("/")[-1])
        self.pos = 0
        self.cache = {}
        self.bytes_fetched = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        if whence == 0:
            self.pos = off
        elif whence == 1:
            self.pos += off
        else:
            self.pos = self.size + off
        return self.pos

    def _block(self, i):
        if i not in self.cache:
            a = i * BLOCK
            b = min(self.size, a + BLOCK) - 1
            for attempt in range(5):
                try:
                    r = self.s.get(self.url, headers={"Range": f"bytes={a}-{b}"}, timeout=180)
                    if r.status_code == 206:
                        self.cache[i] = r.content
                        self.bytes_fetched += len(r.content)
                        break
                except requests.RequestException:
                    pass
            else:
                raise IOError(f"failed block {i}")
            if len(self.cache) > 256:  # keep memory bounded
                self.cache.pop(next(iter(self.cache)))
        return self.cache[i]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while n > 0:
            i, off = divmod(self.pos, BLOCK)
            blk = self._block(i)
            chunk = blk[off:off + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def main():
    mode, url = sys.argv[1], sys.argv[2]
    f = HTTPRangeFile(url)
    z = zipfile.ZipFile(f)
    rx = re.compile(sys.argv[3]) if len(sys.argv) > 3 else None
    members = [m for m in z.infolist() if not m.is_dir() and (rx is None or rx.search(m.filename))]
    if mode == "list":
        tot = 0
        for m in members:
            print(f"{m.file_size:>12} {m.compress_size:>12} {m.filename}")
            tot += m.compress_size
        print("members:", len(members), "compressed bytes:", tot, "archive size:", f.size)
        return
    out = Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    for m in members:
        parts = Path(m.filename).parts
        tag = re.sub(r"[^A-Za-z0-9._+-]", "_", "__".join(parts[-3:-1])) if len(parts) > 1 else ""
        name = re.sub(r"[^A-Za-z0-9._+()-]", "_", parts[-1])
        dest = out / (f"{tag}__{name}" if tag else name)
        dest.write_bytes(z.read(m))
        print("extracted", m.filename, "->", dest.name, m.file_size, flush=True)
    print("bytes fetched:", f.bytes_fetched)


if __name__ == "__main__":
    main()
