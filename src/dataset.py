# %% Imports

from collections.abc import Callable
from functools import cache
from math import ceil
from pathlib import Path
from typing import NamedTuple

import soundfile
from torch import Generator, Tensor, from_numpy, rand, randint, randperm
from torch.nn.functional import pad
from torch.utils.data import Dataset

# %% Global Variables

LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "_unknown_", "_silence_"]

LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}

UNKNOWN_IDX = LABEL_TO_IDX["_unknown_"]
SILENCE_IDX = LABEL_TO_IDX["_silence_"]

SAMPLE_RATE = 16000

CLIP_SAMPLES = 16000

UNKNOWN_FRACTION = 0.10
SILENCE_FRACTION = 0.10

DATA_DIR = Path(Path.cwd().parent / "data/raw")
DATASET_DIR = DATA_DIR / "speech_commands_v0.02"
TEST_SET_DIR = DATA_DIR / "speech_commands_test_set_v0.02"
BACKGROUND_NOISE_DIR = DATASET_DIR / "_background_noise_"

# %% Loading the Dataset

def read_split_list(name: str) -> set[str]:
    return set((DATASET_DIR / name).read_text().split())

class Clip(NamedTuple):
    path: Path
    word: str
    label: int
    speaker: str
    utterance: int

def list_clips() -> dict[str, list[Clip]]:
    validation = read_split_list("validation_list.txt")
    testing = read_split_list("testing_list.txt")
    splits: dict[str, list[Clip]] = {"train": [], "val": [], "test": []}

    for word_dir in sorted(DATASET_DIR.iterdir()):
        if not word_dir.is_dir() or word_dir == BACKGROUND_NOISE_DIR:
            continue
        
        word = word_dir.name
        label = LABEL_TO_IDX.get(word, UNKNOWN_IDX)
        
        for path in sorted(word_dir.glob("*.wav")):
            key = f"{word}/{path.name}"
            
            if key in validation:
                split = "val"
            elif key in testing:
                split = "test"
            else:
                split = "train"

            # each person (speaker) records the same word n times in the dataset
            speaker, n = path.stem.split("_nohash_")
            splits[split].append(Clip(path, word, label, speaker, int(n)))

    return splits

# %% Turning the audios into Tensors

def load_clip(path: Path) -> Tensor:
    samples, _ = soundfile.read(path, dtype="float32")
    waveform = from_numpy(samples[:CLIP_SAMPLES])
    return pad(waveform, (0, CLIP_SAMPLES - len(waveform)))

# %% Getting Tensors from Background Noise

@cache
def background_noise() -> list[Tensor]:
    paths = sorted(BACKGROUND_NOISE_DIR.glob("*.wav"))
    return [from_numpy(soundfile.read(path, dtype="float32")[0]) for path in paths]

# %% Dataset

class SpeechCommands(Dataset):
    def __init__(self, split: str, transform: Callable[[Tensor], Tensor] | None = None, seed: int = 0):
        if split != "train" and split != "val":
            raise ValueError(f"Split != than train or val, value received: '{split}'")

        clips = list_clips()[split]
        keyword_share = 1 - UNKNOWN_FRACTION - SILENCE_FRACTION # 0.8 == 80%

        self.split = split
        self.transform = transform
        self.seed = seed
        self.keywords = [clip for clip in clips if clip.label != UNKNOWN_IDX]
        self.auxiliary = [clip for clip in clips if clip.label == UNKNOWN_IDX]
        self.n_keyword = len(self.keywords)
        self.n_unknown = ceil(round((self.n_keyword * UNKNOWN_FRACTION)/keyword_share, 6))
        self.n_silence = ceil(round((self.n_keyword * SILENCE_FRACTION)/keyword_share, 6))
        self.draw = 0

        self.resample()

    def resample(self):
        if self.split == "val" and self.draw > 0:
            return

        generator = Generator().manual_seed(self.seed * 2**10 + self.draw)
        self.draw += 1

        permutation = randperm(len(self.auxiliary), generator=generator)
        self.unknown_indices = permutation[:self.n_unknown].tolist()

        tracks = background_noise()
        offsets = rand(self.n_silence, generator=generator).tolist()
        self.silence_tracks = randint(len(tracks), (self.n_silence,), generator=generator).tolist()
        self.silence_offsets = [
            int(offset * (len(tracks[track]) - CLIP_SAMPLES + 1))
            for track, offset in zip(self.silence_tracks, offsets)
        ]
        self.silence_gains = rand(self.n_silence, generator=generator).tolist()

    def __len__(self) -> int:
        return self.n_keyword + self.n_unknown + self.n_silence

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        if idx < 0:
            raise IndexError("Negative index in __getitem__ from SpeechCommands")
        
        if idx < self.n_keyword:
            clip = self.keywords[idx]
        elif idx < self.n_keyword + self.n_unknown:
            clip = self.auxiliary[self.unknown_indices[idx - self.n_keyword]]
        else:
            window = idx - self.n_keyword - self.n_unknown
            track = background_noise()[self.silence_tracks[window]]
            offset = self.silence_offsets[window]
            return track[offset:offset + CLIP_SAMPLES] * self.silence_gains[window], SILENCE_IDX

        waveform = load_clip(clip.path)

        if self.transform is not None:
            waveform = self.transform(waveform)

        return waveform, clip.label

class SpeechCommandsTest(Dataset):
    def __init__(self):
        self.files: list[Path] = []
        self.labels: list[int] = []

        for label_dir in sorted(TEST_SET_DIR.iterdir()):
            if not label_dir.is_dir():
                continue

            label = LABEL_TO_IDX[label_dir.name]
            files = sorted(label_dir.glob("*.wav"))

            self.files += files
            self.labels += [label] * len(files)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        return load_clip(self.files[index]), self.labels[index]
