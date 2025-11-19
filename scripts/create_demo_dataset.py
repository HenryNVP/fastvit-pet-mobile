from __future__ import annotations

import csv
import shutil
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
SOURCE_SPLITS = OrderedDict(
    {
        "train": {
            "images": ROOT_DIR / "data" / "train",
            "labels": ROOT_DIR / "data" / "train_labels.csv",
            "per_class": 20,
        },
        "val": {
            "images": ROOT_DIR / "data" / "val",
            "labels": ROOT_DIR / "data" / "val_labels.csv",
            "per_class": 5,
        },
        "test": {
            "images": ROOT_DIR / "data" / "test",
            "labels": ROOT_DIR / "data" / "test_labels.csv",
            "per_class": 5,
        },
    }
)
DEMO_ROOT = ROOT_DIR / "demo"


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def select_classes(rows: Sequence[Dict[str, str]], limit: int) -> List[str]:
    ordered: List[str] = []
    for row in rows:
        cls = row["class_name"]
        if cls not in ordered:
            ordered.append(cls)
            if len(ordered) == limit:
                break
    if len(ordered) < limit:
        raise ValueError(f"Only found {len(ordered)} classes; expected {limit}.")
    return ordered


def sample_rows(
    rows: Iterable[Dict[str, str]],
    classes: Sequence[str],
    per_class: int,
) -> List[Dict[str, str]]:
    buckets: Dict[str, List[Dict[str, str]]] = {cls: [] for cls in classes}
    for row in rows:
        cls = row["class_name"]
        if cls not in buckets:
            continue
        if len(buckets[cls]) >= per_class:
            continue
        buckets[cls].append(row)
        if all(len(bucket) >= per_class for bucket in buckets.values()):
            break

    missing = [cls for cls, bucket in buckets.items() if len(bucket) < per_class]
    if missing:
        raise ValueError(
            f"Not enough samples for {missing}; need {per_class} per class."
        )

    result: List[Dict[str, str]] = []
    for cls in classes:
        result.extend(buckets[cls])
    return result


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_images(rows: Iterable[Dict[str, str]], src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        image_name = row["image"]
        src = src_dir / image_name
        if not src.exists():
            raise FileNotFoundError(f"Image missing: {src}")
        shutil.copy2(src, dst_dir / image_name)


def write_labels(rows: Iterable[Dict[str, str]], path: Path) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image", "class_name", "class_id", "species", "breed_id"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    train_rows = load_rows(SOURCE_SPLITS["train"]["labels"])
    selected_classes = select_classes(train_rows, limit=2)
    print(f"Selected classes: {selected_classes}")

    ensure_clean_dir(DEMO_ROOT)

    for split_name, cfg in SOURCE_SPLITS.items():
        rows = load_rows(cfg["labels"])
        sampled = sample_rows(rows, selected_classes, cfg["per_class"])

        split_dir = DEMO_ROOT / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        copy_images(sampled, cfg["images"], split_dir)
        write_labels(sampled, DEMO_ROOT / f"{split_name}_labels.csv")
        print(
            f"{split_name}: copied {len(sampled)} images "
            f"({cfg['per_class']} per class)"
        )

    print(f"Demo dataset created under {DEMO_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
