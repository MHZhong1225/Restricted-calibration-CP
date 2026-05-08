import argparse
import glob
import os
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm import tqdm


def _collect_paths(source_dir: Path, classes: list[str], exts: list[str]):
    data = []
    for class_name in classes:
        class_dir = source_dir / class_name
        if not class_dir.is_dir():
            continue
        class_paths = []
        for ext in exts:
            class_paths.extend(glob.glob(str(class_dir / f"*.{ext}")))
        for p in sorted(set(class_paths)):
            data.append((p, class_name))
    return data


def _link_or_copy(src: str, dst_dir: Path, mode: str):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / Path(src).name
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        os.symlink(os.path.abspath(src), dst)
        return
    raise ValueError(f"Unknown mode={mode!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--target-dir", default="dataset/bach")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layout", choices=["flat", "split"], default="flat")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--exts", default="tif,tiff,png,jpg,jpeg")
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source_dir = Path(args.source_dir).expanduser().resolve()
    target_dir = Path(args.target_dir).expanduser().resolve()

    if not source_dir.is_dir():
        raise FileNotFoundError(f"source-dir not found: {source_dir}")

    train_frac = float(args.train_frac)
    val_frac = float(args.val_frac)
    test_frac = float(args.test_frac)
    total_frac = train_frac + val_frac + test_frac
    if abs(total_frac - 1.0) > 1e-8:
        raise ValueError(f"train/val/test fractions must sum to 1.0, got {total_frac}")
    if min(train_frac, val_frac, test_frac) <= 0:
        raise ValueError("train/val/test fractions must all be > 0")

    classes = ["Benign", "InSitu", "Invasive", "Normal"]
    exts = [e.strip().lstrip(".") for e in str(args.exts).split(",") if e.strip()]

    data = _collect_paths(source_dir, classes=classes, exts=exts)
    if len(data) == 0:
        raise RuntimeError(f"No images found under {source_dir} for classes={classes} exts={exts}")

    if target_dir.exists():
        if not args.force:
            raise FileExistsError(f"target-dir already exists: {target_dir} (use --force to overwrite)")
        shutil.rmtree(target_dir)

    if args.layout == "flat":
        for src, cls in tqdm(data, desc="all"):
            _link_or_copy(src, target_dir / cls, mode=args.mode)
        print(f"source_dir={source_dir}")
        print(f"target_dir={target_dir}")
        print(f"total={len(data)}")
        return

    labels = [lab for _p, lab in data]
    tmp_frac = val_frac + test_frac
    train_data, tmp_data = train_test_split(
        data,
        test_size=tmp_frac,
        random_state=int(args.seed),
        stratify=labels,
    )
    if len(tmp_data) == 0:
        raise RuntimeError("Split produced empty val/test. Check fractions and dataset size.")
    val_relative = val_frac / tmp_frac
    val_data, test_data = train_test_split(
        tmp_data,
        test_size=(1.0 - val_relative),
        random_state=int(args.seed),
        stratify=[lab for _p, lab in tmp_data],
    )

    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        for src, cls in tqdm(split_data, desc=f"{split_name}"):
            _link_or_copy(src, target_dir / split_name / cls, mode=args.mode)

    print(f"source_dir={source_dir}")
    print(f"target_dir={target_dir}")
    print(f"total={len(data)} train={len(train_data)} val={len(val_data)} test={len(test_data)}")


if __name__ == "__main__":
    main()
