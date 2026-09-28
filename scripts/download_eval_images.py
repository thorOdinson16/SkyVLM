import io
import struct
import threading
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

BASE_URL = "https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript"

IMAGE_DIR = ROOT / "dataset" / "images"
MANIFEST_DIR = ROOT / "dataset" / "manifests"

SPLITS = {
    "val_5k": ROOT / "SkyScript_val_5K_filtered_by_CLIP_laion_RS.csv",
    "test_30k": ROOT / "SkyScript_test_30K_filtered_by_CLIP_laion_RS.csv",
}

# Same caption field used for train_200k.csv
CAPTION_COLUMN = "title_multi_objects"

NUM_WORKERS = 32
RETRIES = 5

LOCAL_HEADER_SIZE = 30

# Local headers may carry a longer extra field than the central directory.
EXTRA_SLACK = 1024


# ---------------------------------------------------------
# Remote zip access via HTTP range requests
# ---------------------------------------------------------

_thread_local = threading.local()


def session():
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


def fetch_range(url, start, end):
    for attempt in range(RETRIES):
        try:
            response = session().get(
                url,
                headers={"Range": f"bytes={start}-{end}"},
                timeout=60,
            )
            response.raise_for_status()
            return response.content
        except requests.RequestException:
            if attempt == RETRIES - 1:
                raise


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file backed by HTTP range requests."""

    def __init__(self, url):
        self.url = url
        self.size = int(requests.head(url, timeout=60).headers["Content-Length"])
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        data = fetch_range(self.url, self.pos, self.pos + n - 1)
        self.pos += len(data)
        return data


def read_central_directory(archive):
    url = f"{BASE_URL}/{archive}.zip"
    with zipfile.ZipFile(HTTPRangeFile(url)) as zf:
        return url, {info.filename: info for info in zf.infolist()}


def download_member(url, info, destination):
    start = info.header_offset
    end = (
        start
        + LOCAL_HEADER_SIZE
        + len(info.filename.encode())
        + EXTRA_SLACK
        + info.compress_size
    )
    buf = fetch_range(url, start, end)

    if buf[:4] != b"PK\x03\x04":
        raise ValueError(f"Bad local header: {info.filename}")

    name_len, extra_len = struct.unpack("<HH", buf[26:30])
    data_start = LOCAL_HEADER_SIZE + name_len + extra_len
    data = buf[data_start:data_start + info.compress_size]

    if info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompressobj(-15).decompress(data)
    elif info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"Unsupported compression: {info.filename}")

    if zlib.crc32(data) != info.CRC:
        raise ValueError(f"CRC mismatch: {info.filename}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(destination)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    print("=" * 70)
    print("SkyScript Val/Test Image Downloader")
    print("=" * 70)

    splits = {
        name: pd.read_csv(path)
        for name, path in SPLITS.items()
    }

    filepaths = sorted(
        set().union(*(df["filepath"] for df in splits.values()))
    )

    todo = [
        p for p in filepaths
        if not (IMAGE_DIR / p).exists()
    ]

    print(f"Images required : {len(filepaths):,}")
    print(f"Already on disk : {len(filepaths) - len(todo):,}")
    print(f"To download     : {len(todo):,}")

    by_archive = {}
    for p in todo:
        archive, _, name = p.partition("/")
        by_archive.setdefault(archive, []).append(name)

    failures = []

    for archive, names in sorted(by_archive.items()):
        print(f"\nReading central directory: {archive}.zip")
        url, index = read_central_directory(archive)

        # Match on basename; archives may or may not nest a top-level folder.
        by_basename = {
            Path(key).name: info
            for key, info in index.items()
            if not info.is_dir()
        }

        missing = [n for n in names if n not in by_basename]
        if missing:
            print(f"  Not found in archive: {len(missing)}")
            failures += [f"{archive}/{n}" for n in missing]

        with ThreadPoolExecutor(NUM_WORKERS) as pool:
            futures = {
                pool.submit(
                    download_member,
                    url,
                    by_basename[n],
                    IMAGE_DIR / archive / n,
                ): f"{archive}/{n}"
                for n in names
                if n in by_basename
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=archive,
                unit="img",
            ):
                try:
                    future.result()
                except Exception as error:
                    failures.append(futures[future])
                    tqdm.write(f"  FAILED {futures[future]}: {error}")

    # -----------------------------------------------------
    # Verification
    # -----------------------------------------------------

    print("\nVerifying images...")
    bad = []
    for p in tqdm(filepaths, unit="img"):
        try:
            with Image.open(IMAGE_DIR / p) as image:
                image.verify()
        except Exception:
            bad.append(p)

    print(f"Valid images   : {len(filepaths) - len(bad):,} / {len(filepaths):,}")
    print(f"Download fails : {len(failures):,}")

    if bad:
        print("First invalid:", bad[:10])
        raise SystemExit("Verification failed; re-run to retry missing files.")

    # -----------------------------------------------------
    # Manifests (same format as train_200k.csv)
    # -----------------------------------------------------

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    for name, df in splits.items():
        out = MANIFEST_DIR / f"{name}.csv"
        pd.DataFrame({
            "image_path": df["filepath"],
            "caption": df[CAPTION_COLUMN],
        }).to_csv(out, index=False)
        print(f"Wrote {out} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
