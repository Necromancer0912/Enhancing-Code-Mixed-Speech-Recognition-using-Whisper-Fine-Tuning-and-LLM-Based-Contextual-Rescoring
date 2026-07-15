"""Download the MUCS Bengali-English archives from OpenSLR-104.

The OpenSLR mirrors throttle single connections hard, so the file is split
into ranged chunks fetched concurrently across all three mirrors and then
stitched back together. Resumable: finished chunks are kept on disk.

Usage:
    python scripts/download_data.py
"""
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ROOT

MIRRORS = [
    "https://openslr.elda.org/resources/104/",
    "https://www.openslr.org/resources/104/",
    "https://openslr.trmal.net/resources/104/",
]
FILES = ["Bengali-English_test.tar.gz", "Bengali-English_train.tar.gz"]

CHUNK = 32 * 1024 * 1024
WORKERS = 12

print_lock = threading.Lock()


def remote_size(fname):
    req = urllib.request.Request(MIRRORS[0] + fname, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers["Content-Length"])


def fetch_range(fname, start, end, mirror, out_path):
    req = urllib.request.Request(mirror + fname, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as r, open(out_path, "wb") as f:
        while True:
            buf = r.read(256 * 1024)
            if not buf:
                break
            f.write(buf)


def download(fname, dest_dir):
    total = remote_size(fname)
    n_chunks = (total + CHUNK - 1) // CHUNK
    print(f"{fname}: {total / 1e9:.2f} GB in {n_chunks} chunks")

    target = dest_dir / fname
    if target.exists() and target.stat().st_size == total:
        print("  already downloaded, skipping")
        return

    parts_dir = dest_dir / (fname + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)

    todo = []
    for i in range(n_chunks):
        start = i * CHUNK
        end = min(total, (i + 1) * CHUNK) - 1
        part = parts_dir / f"{i:05d}.part"
        if not (part.exists() and part.stat().st_size == end - start + 1):
            todo.append((i, start, end, part))

    done = [n_chunks - len(todo)]
    t0 = time.time()
    it = iter(todo)
    it_lock = threading.Lock()

    def worker(wid):
        mirror = MIRRORS[wid % len(MIRRORS)]
        while True:
            with it_lock:
                try:
                    i, start, end, part = next(it)
                except StopIteration:
                    return
            for attempt in range(5):
                try:
                    m = mirror if attempt < 3 else MIRRORS[attempt % len(MIRRORS)]
                    fetch_range(fname, start, end, m, part)
                    break
                except Exception as e:
                    with print_lock:
                        print(f"  retry chunk {i}: {e}")
                    time.sleep(2 * (attempt + 1))
            with print_lock:
                done[0] += 1
                mb = done[0] * CHUNK / 1e6
                el = max(time.time() - t0, 1e-9)
                print(f"  {done[0]}/{n_chunks} chunks, {mb / el:.1f} MB/s avg")

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    with open(target, "wb") as out:
        for i in range(n_chunks):
            part = parts_dir / f"{i:05d}.part"
            out.write(part.read_bytes())
    assert target.stat().st_size == total, "size mismatch after stitching"

    for i in range(n_chunks):
        os.remove(parts_dir / f"{i:05d}.part")
    parts_dir.rmdir()
    print(f"  done: {target}")


def main():
    dest = ROOT / "Data" / "downloads"
    dest.mkdir(parents=True, exist_ok=True)
    for fname in FILES:
        download(fname, dest)


if __name__ == "__main__":
    main()
