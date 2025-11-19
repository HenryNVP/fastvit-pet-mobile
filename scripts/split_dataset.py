from __future__ import annotations

import csv
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
IMAGES_DIR = RAW_DIR / "images"
ANNOTATIONS_DIR = RAW_DIR / "annotations"
TRAINVAL_LIST = ANNOTATIONS_DIR / "trainval.txt"
TEST_LIST = ANNOTATIONS_DIR / "test.txt"

TRAIN_DIR = ROOT_DIR / "data" / "train"
VAL_DIR = ROOT_DIR / "data" / "val"
TEST_DIR = ROOT_DIR / "data" / "test"

TRAIN_LABELS = ROOT_DIR / "data" / "train_labels.csv"
VAL_LABELS = ROOT_DIR / "data" / "val_labels.csv"
TEST_LABELS = ROOT_DIR / "data" / "test_labels.csv"

IMAGE_SIZE = (256, 256)


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_split(file_path: Path) -> List[str]:
    names: List[str] = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            names.append(raw.split()[0])
    return names


def read_metadata() -> Dict[str, Dict[str, str]]:
    labels: Dict[str, Dict[str, str]] = {}
    with open(ANNOTATIONS_DIR / "list.txt", "r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw or raw.startswith("#"):
                continue
            parts = raw.strip().split()
            if len(parts) < 4:
                continue
            name, class_id, species, breed_id = parts[:4]
            labels[name] = {
                "class_id": class_id,
                "species": species,
                "breed_id": breed_id,
                "class_name": name.rsplit("_", 1)[0],
            }
    return labels


def resize_and_copy(image_name: str, destination: Path) -> None:
    src = IMAGES_DIR / f"{image_name}.jpg"
    if not src.exists():
        png = IMAGES_DIR / f"{image_name}.png"
        if png.exists():
            src = png
        else:
            raise FileNotFoundError(f"Missing source image for {image_name}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = img.convert("RGB").resize(IMAGE_SIZE, Image.BILINEAR)
        img.save(destination, format="JPEG", quality=95)


def write_labels(names: Iterable[str], metadata: Dict[str, Dict[str, str]], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "class_name", "class_id", "species", "breed_id"])
        for name in names:
            info = metadata[name]
            writer.writerow(
                [
                    f"{name}.jpg",
                    info["class_name"],
                    info["class_id"],
                    info["species"],
                    info["breed_id"],
                ]
            )


def populate_split(names: Iterable[str], metadata: Dict[str, Dict[str, str]], directory: Path) -> None:
    for name in names:
        resize_and_copy(name, directory / f"{name}.jpg")


def split_test(names: List[str], metadata: Dict[str, Dict[str, str]]) -> Tuple[List[str], List[str]]:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for name in names:
        grouped[metadata[name]["class_name"]].append(name)

    val_split: List[str] = []
    remaining_test: List[str] = []

    for class_names, entries in grouped.items():
        entries.sort()
        mid = len(entries) // 2
        val_split.extend(entries[:mid])
        remaining_test.extend(entries[mid:])

    return val_split, remaining_test


def main() -> int:
    metadata = read_metadata()

    train_names = read_split(TRAINVAL_LIST)
    test_names = read_split(TEST_LIST)
    val_names, remaining_test_names = split_test(test_names, metadata)

    ensure_clean_dir(TRAIN_DIR)
    ensure_clean_dir(VAL_DIR)
    ensure_clean_dir(TEST_DIR)

    populate_split(train_names, metadata, TRAIN_DIR)
    populate_split(val_names, metadata, VAL_DIR)
    populate_split(remaining_test_names, metadata, TEST_DIR)

    write_labels(train_names, metadata, TRAIN_LABELS)
    write_labels(val_names, metadata, VAL_LABELS)
    write_labels(remaining_test_names, metadata, TEST_LABELS)

    print(
        f"Prepared train={len(train_names)} val={len(val_names)} test={len(remaining_test_names)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
