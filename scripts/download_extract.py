"""Download MARCO TFRecord shards and extract images + a metadata manifest.

Only pulls from the `train-jpg` directory on marco.ccr.buffalo.edu (407 shards, ~1024 images
each). That alone is more than enough for our working dataset - see docs/ and experiments/ for
how many shards we actually use. This is unrelated to our own train/val/test split (data/splits.csv),
which is built separately by scripts/make_splits.py.
"""

import argparse
import urllib.request
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from tfrecord.reader import tfrecord_loader

SHARD_BASE_URL = "https://marco.ccr.buffalo.edu/data/train-jpg"
TOTAL_SHARDS = 407


def shard_name(shard_index: int) -> str:
    """Build the MARCO shard filename, e.g. train-00100-of-00407."""
    return f"train-{shard_index:05d}-of-{TOTAL_SHARDS:05d}"


def pick_shard_indices(n_shards: int) -> list[int]:
    """Pick n_shards indices evenly spaced across the full 1..TOTAL_SHARDS range."""
    if n_shards >= TOTAL_SHARDS:
        return list(range(1, TOTAL_SHARDS + 1))
    positions = np.linspace(1, TOTAL_SHARDS, num=n_shards)
    return sorted({int(round(p)) for p in positions})


def download_shard(shard_index: int, cache_dir: Path) -> Path:
    """Download one shard into cache_dir, skipping it if already present."""
    name = shard_name(shard_index)
    path = cache_dir / name
    if not path.exists():
        urllib.request.urlretrieve(f"{SHARD_BASE_URL}/{name}", path)
    return path


def iter_shard_records(shard_path: Path) -> Iterator[dict]:
    """Yield decoded records from a shard.

    Scalar numeric TFRecord fields come back from `tfrecord_loader` as 1-element numpy
    arrays; byte/string fields come back as plain `bytes`, not list-wrapped - indexing a
    bytes field with [0] silently grabs its first raw byte instead of the whole value.
    """
    for record in tfrecord_loader(str(shard_path), None):
        yield {
            "filename": record["image/filename"].decode("utf-8"),
            "image_id": int(record["image/id"][0]),
            "label": record["image/class/text"].decode("utf-8"),
            "source": int(record["image/class/source"][0]),
            "jpeg_bytes": record["image/encoded"],
        }


def extract_shard(shard_path: Path, out_dir: Path, shard_index: int) -> list[dict]:
    """Write every image in a shard to out_dir/<label>/<filename>, return manifest rows."""
    rows = []
    for record in iter_shard_records(shard_path):
        class_dir = out_dir / record["label"]
        class_dir.mkdir(parents=True, exist_ok=True)
        image_path = class_dir / record["filename"]
        image_path.write_bytes(record["jpeg_bytes"])
        rows.append(
            {
                "shard": shard_index,
                "filename": record["filename"],
                "image_id": record["image_id"],
                "label": record["label"],
                "source": record["source"],
                "path": str(image_path.relative_to(out_dir.parent)),
                "n_bytes": len(record["jpeg_bytes"]),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-shards", type=int, default=15,
        help="Number of shards to pull, evenly spaced across the full range (default: 15).",
    )
    parser.add_argument(
        "--shards", type=int, nargs="+", default=None,
        help="Explicit shard indices to pull, overrides --n-shards.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/_shards"))
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/manifest.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    shard_indices = args.shards if args.shards else pick_shard_indices(args.n_shards)
    print(f"Pulling {len(shard_indices)} shards: {shard_indices}")

    rows = []
    for idx in shard_indices:
        shard_path = download_shard(idx, args.cache_dir)
        rows.extend(extract_shard(shard_path, args.out_dir, idx))

    manifest_df = pd.DataFrame(rows)
    manifest_df.to_csv(args.manifest, index=False)

    print(f"Extracted {len(manifest_df)} images to {args.out_dir}")
    print(manifest_df["label"].value_counts())
    print(f"Manifest written to {args.manifest}")


if __name__ == "__main__":
    main()
