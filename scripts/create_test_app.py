from __future__ import annotations

import csv
import shutil
from pathlib import Path

from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
TEST_DIR = DATA_DIR / "test"
TEST_LABELS = DATA_DIR / "test_labels.csv"

TEST_APP_DIR = DATA_DIR / "test_app"
TEST_APP_IMAGES_DIR = TEST_APP_DIR / "images"
TEST_APP_LABELS = TEST_APP_DIR / "test.txt"

IMAGE_SIZE = (256, 256)
EXPECTED_TEST_SIZE = 1837


def ensure_clean_dir(path: Path) -> None:
    """Create an empty directory, removing it first if it already exists."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_test_images() -> int:
    """
    Copy all images from data/test into data/test_app/images as a flat
    directory (no class subfolders).
    """
    count = 0
    for src in TEST_DIR.rglob("*.jpg"):
        # Use only the file name to flatten structure, e.g.
        # "Abyssinian/Abyssinian_51.jpg" -> "Abyssinian_51.jpg"
        dst = TEST_APP_IMAGES_DIR / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)

        # Copy bytes directly to avoid any transformation
        shutil.copy2(src, dst)
        count += 1
    return count


def copy_test_labels() -> int:
    """
    Read the CSV annotations for test images and write them out
    as a plain text annotation file (no header) compatible with
    the original annotation format.
    Returns the number of data rows written.
    """
    if not TEST_LABELS.exists():
        raise FileNotFoundError(f"Missing test labels file at {TEST_LABELS}")

    row_count = 0
    with open(TEST_LABELS, "r", encoding="utf-8", newline="") as in_f, open(
        TEST_APP_LABELS, "w", encoding="utf-8", newline=""
    ) as out_f:
        reader = csv.reader(in_f)
        # Skip header: ["image","class_name","class_id","species","breed_id"]
        header = next(reader, None)
        _ = header  # silence type checkers

        for image_path, class_name, class_id, species, breed_id in reader:
            # image_path looks like "Abyssinian/Abyssinian_51.jpg"
            # We want the base name without extension, e.g. "Abyssinian_51"
            stem = Path(image_path).stem
            # Write in the same numeric format as original annotations:
            # "<name> <class_id> <species> <breed_id>"
            out_f.write(f"{stem} {class_id} {species} {breed_id}\n")
            row_count += 1

    return row_count


def main() -> int:
    # Prepare clean output directory
    ensure_clean_dir(TEST_APP_DIR)
    TEST_APP_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    image_count = copy_test_images()
    label_count = copy_test_labels()

    if label_count != EXPECTED_TEST_SIZE:
        print(
            f"Warning: annotation rows ({label_count}) "
            f"do not match expected test size ({EXPECTED_TEST_SIZE})."
        )

    if image_count != label_count:
        print(
            f"Warning: image count ({image_count}) "
            f"does not match number of annotations ({label_count})."
        )

    print(
        f"Created test_app with {image_count} images and "
        f"{label_count} text annotations in {TEST_APP_DIR}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


