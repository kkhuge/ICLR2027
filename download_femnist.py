"""Download the official TFF Federated EMNIST archive and extract its HDF5 files.

Only the two expected HDF5 members are extracted. The dataset itself is not
redistributed in this repository.
"""

import shutil
import tarfile
import urllib.request
from pathlib import Path


URL = "https://storage.googleapis.com/tff-datasets-public/fed_emnist.tar.bz2"
RAW_DIR = Path(__file__).resolve().parent / "data" / "femnist" / "raw"
NAMES = ("fed_emnist_train.h5", "fed_emnist_test.h5")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if all((RAW_DIR / name).exists() for name in NAMES):
        print("Official TFF FEMNIST HDF5 files already present.")
        return

    archive_path = RAW_DIR / "fed_emnist.tar.bz2"
    if not archive_path.exists():
        print(f"Downloading {URL}", flush=True)
        urllib.request.urlretrieve(URL, archive_path)

    with tarfile.open(archive_path, "r:bz2") as archive:
        members = {Path(member.name).name: member for member in archive.getmembers()}
        for name in NAMES:
            if name not in members or not members[name].isfile():
                raise FileNotFoundError(f"Archive does not contain {name}")
            target = RAW_DIR / name
            if target.exists():
                continue
            source = archive.extractfile(members[name])
            if source is None:
                raise OSError(f"Could not read {name} from archive")
            temporary = target.with_suffix(target.suffix + ".tmp")
            with source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            temporary.replace(target)
            print(f"Extracted {target}", flush=True)
    archive_path.unlink()


if __name__ == "__main__":
    main()
