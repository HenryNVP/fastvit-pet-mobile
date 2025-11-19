from __future__ import annotations

from pathlib import Path
from urllib import request

import tarfile
import zipfile


ROOT_DIR = Path(__file__).resolve().parents[1]
TARGET_DIR = ROOT_DIR / "data" / "raw"
DATASETS = [
    "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz",
    "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz",
]


def download_and_extract(destination: Path = TARGET_DIR) -> Path:
    destination.mkdir(parents=True, exist_ok=True)

    for url in DATASETS:
        filename = url.rstrip("/").split("/")[-1] or "downloaded_file"
        archive_path = destination / filename

        if archive_path.exists():
            print(f"Using existing file at {archive_path}")
        else:
            print(f"Downloading {filename} from {url}")
            request.urlretrieve(url, archive_path)

        if tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, "r:*") as tf:
                tf.extractall(destination)
            continue

        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(destination)
            continue

        print(f"Skipped extraction for {archive_path.name}: not a tar or zip archive.")

    return destination


if __name__ == "__main__":
    download_and_extract()
