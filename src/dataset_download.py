# %% Imports

import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

# %% Global Variables

_CITATION = """
@article{speechcommandsv2,
   author = { {Warden}, P.},
    title = "{Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition}",
  journal = {ArXiv e-prints},
  archivePrefix = "arXiv",
  eprint = {1804.03209},
  primaryClass = "cs.CL",
  keywords = {Computer Science - Computation and Language, Computer Science - Human-Computer Interaction},
    year = 2018,
    month = apr,
    url = {https://arxiv.org/abs/1804.03209},
}
"""

DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

TEST_SET_URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz"

DATA_DIR = Path(Path.cwd().parent / "data/raw")
DATASET_DIR = DATA_DIR / "speech_commands_v0.02"
TEST_SET_DIR = DATA_DIR / "speech_commands_test_set_v0.02"

# (url, extraction directory, path whose existence means the archive is already extracted)
ARCHIVES = [
    (DATASET_URL, DATASET_DIR, DATASET_DIR / "validation_list.txt"),
    (TEST_SET_URL, TEST_SET_DIR, TEST_SET_DIR / "_silence_"),
]

# %% Acquisition

DATA_DIR.mkdir(parents=True, exist_ok=True)

for url, dataset_dir, marker in ARCHIVES:
    if marker.exists():
        print(f"Dataset already at {dataset_dir}")
        continue

    archive_path = DATA_DIR / url.rsplit("/", 1)[1]
    with urllib.request.urlopen(url) as response, open(archive_path, "wb") as archive:
        total = int(response.headers.get("Content-Length", 0)) or None
        with tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {archive_path.name}") as bar:
            while chunk := response.read(1024 * 1024):
                archive.write(chunk)
                bar.update(len(chunk))

    dataset_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(dataset_dir, filter="data")

    print(f"Dataset acquired at {dataset_dir}")
