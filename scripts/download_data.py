"""Download the official HetRec 2011 Last.fm 2K dataset."""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path


DATA_URL = "https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"


def download_data(project_dir: Path) -> Path:
    raw_dir = project_dir.resolve() / "data" / "raw"
    data_dir = raw_dir / "lastfm-2k"
    if (data_dir / "user_artists.dat").exists() and (data_dir / "artists.dat").exists():
        return data_dir

    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / "hetrec2011-lastfm-2k.zip"
    if not archive_path.exists():
        partial_path = archive_path.with_suffix(".zip.part")
        urllib.request.urlretrieve(DATA_URL, partial_path)
        partial_path.replace(archive_path)

    data_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            target = (data_dir / entry.filename).resolve()
            if not target.is_relative_to(data_dir.resolve()):
                raise ValueError(f"Unexpected archive path: {entry.filename}")
        archive.extractall(data_dir)

    if not (data_dir / "user_artists.dat").exists() or not (data_dir / "artists.dat").exists():
        raise FileNotFoundError("The archive does not contain the expected Last.fm data files.")
    return data_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(f"Dataset ready: {download_data(args.project_dir)}")
    print("See the dataset readme.txt for source credits and usage terms.")


if __name__ == "__main__":
    main()
