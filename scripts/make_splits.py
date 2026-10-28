"""Build the one fixed train/val/test split for all modelling (CLAUDE.md rule 1).

Groups images from the same physical drop (same plate+well, or a detected near-duplicate) so
no group's images cross a split boundary (rule 3), then does a stratified group split so class
proportions stay close to balanced across splits.

Run once. Refuses to overwrite an existing data/splits.csv unless --force is passed, since the
split must never be regenerated once modelling starts.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42

PATTERN_PLATE_WELL = re.compile(r"plateID(\d+).*wellNum(\d+)")
PATTERN_HASH_WELL = re.compile(r"^([0-9a-z]+)_([A-H]\d{1,2})_")


def filename_group_key(filename: str) -> str | None:
    """Group key from filename conventions that encode plate/well directly, else None."""
    match = PATTERN_PLATE_WELL.search(filename)
    if match:
        return f"plate{match.group(1)}well{match.group(2)}"
    match = PATTERN_HASH_WELL.search(filename)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    return None


def average_hash(image_path: Path, hash_size: int = 8) -> str:
    """Cheap near-duplicate signal for images with no filename group key."""
    img = Image.open(image_path).convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    return "".join("1" if p > avg else "0" for p in pixels)


def assign_hash_groups(df: pd.DataFrame, data_dir: Path) -> pd.Series:
    """Group images with no filename key by (hash, label) collisions.

    A hash collision across *different* labels is a false positive (the hash isn't
    discriminative enough for these images) - a true same-drop duplicate must share the
    same ground-truth label. Images with no same-label collision become singleton groups.
    """
    hashes = pd.Series([average_hash(data_dir / p) for p in df["path"]], index=df.index)
    group_keys = pd.Series(index=df.index, dtype=object)
    for (hash_value, label), sub in df.assign(hash=hashes).groupby(["hash", "label"]):
        if len(sub) > 1:
            group_keys.loc[sub.index] = f"hashdup_{hash_value}_{label}"
    ungrouped = group_keys.isna()
    group_keys.loc[ungrouped] = "single_" + df.loc[ungrouped, "image_id"].astype(str)
    return group_keys


def build_group_keys(df: pd.DataFrame, data_dir: Path) -> pd.Series:
    """Assign every image a leakage-group key, filename patterns first, hash fallback second."""
    keys = df["filename"].apply(filename_group_key)
    missing = keys.isna()
    if missing.any():
        keys.loc[missing] = assign_hash_groups(df.loc[missing], data_dir)
    return keys


def group_majority_label(df: pd.DataFrame) -> pd.Series:
    """Majority (mode) label per group, used only to stratify the split.

    A group isn't guaranteed one label: the same plate+well can show different labels across
    imaging timepoints since crystallisation is a kinetic process (documented in docs/).
    """
    mode_per_group = df.groupby("group_key")["label"].agg(lambda s: s.mode().iloc[0])
    return df["group_key"].map(mode_per_group)


def stratified_group_split(
    strat_labels: np.ndarray, groups: np.ndarray, val_frac: float, test_frac: float, seed: int
) -> np.ndarray:
    """Split by group (rule 3) while keeping each group's majority label balanced (rule 1)."""
    n = len(groups)
    placeholder_x = np.zeros(n)

    test_splitter = StratifiedGroupKFold(
        n_splits=round(1 / test_frac), shuffle=True, random_state=seed
    )
    remaining_pos, test_pos = next(test_splitter.split(placeholder_x, strat_labels, groups))

    val_of_remaining = val_frac / (1 - test_frac)
    val_splitter = StratifiedGroupKFold(
        n_splits=round(1 / val_of_remaining), shuffle=True, random_state=seed
    )
    train_pos_rel, val_pos_rel = next(
        val_splitter.split(
            placeholder_x[remaining_pos], strat_labels[remaining_pos], groups[remaining_pos]
        )
    )

    split = np.full(n, "train", dtype=object)
    split[test_pos] = "test"
    split[remaining_pos[val_pos_rel]] = "val"
    split[remaining_pos[train_pos_rel]] = "train"
    return split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/manifest.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--splits-path", type=Path, default=Path("data/splits.csv"))
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing splits file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.splits_path.exists() and not args.force:
        raise SystemExit(
            f"{args.splits_path} already exists - refusing to overwrite (rule 1: the split is "
            "fixed once created). Pass --force if you really mean to regenerate it."
        )

    manifest = pd.read_csv(args.manifest)
    manifest["group_key"] = build_group_keys(manifest, args.data_dir)
    manifest["group_majority_label"] = group_majority_label(manifest)

    manifest["split"] = stratified_group_split(
        manifest["group_majority_label"].to_numpy(),
        manifest["group_key"].to_numpy(),
        args.val_frac,
        args.test_frac,
        args.seed,
    )

    spans = manifest.groupby("group_key")["split"].nunique()
    assert (spans == 1).all(), f"{(spans > 1).sum()} groups span multiple splits - leakage bug"

    out = manifest[["image_id", "filename", "path", "label", "source", "group_key", "split"]]
    out.to_csv(args.splits_path, index=False)

    print(f"seed={args.seed}")
    print(out["split"].value_counts())
    print()
    print(pd.crosstab(out["split"], out["label"]))
    print(f"\nWrote {args.splits_path}")


if __name__ == "__main__":
    main()
