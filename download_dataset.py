"""Download and extract the public University of Sharjah smart-building dataset."""
from pathlib import Path
import zipfile
import requests

URL = "https://zenodo.org/records/12750891/files/Dataset-of-IoT-Based-Energy-and-Environmental-Parameters-in-a-Smart-Building-Infrastructure-2.0.zip?download=1"
RAW_DIR = Path("data/raw")
ARCHIVE = RAW_DIR / "uos_smart_building_v2.zip"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        print("Downloading University of Sharjah dataset...")
        with requests.get(URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            with ARCHIVE.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
    print(f"Saved archive: {ARCHIVE}")

    print("Extracting...")
    with zipfile.ZipFile(ARCHIVE) as z:
        z.extractall(RAW_DIR)
    print("Done. Raw dataset is under data/raw/")


if __name__ == "__main__":
    main()
