"""EuroSAT splits with caller-supplied image preprocessing."""

import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


H5_PATH = Path(__file__).resolve().parents[1] / "data" / "eurosat.h5"


def model_transform(config, training):
    """Resize the entire tile. Archived crop-based runs specify a short edge."""
    size = tuple(config["size"])
    interp = transforms.InterpolationMode(config["interpolation"])
    steps = []
    
    if training:
        steps.extend([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
    
    steps.append(transforms.Resize(config.get("short_edge", size), interpolation=interp, antialias=True))
    
    if "short_edge" in config:
        steps.append(transforms.CenterCrop(size))
    
    steps.extend([transforms.ToTensor(), transforms.Normalize(config["mean"], config["std"])])
    
    return transforms.Compose(steps)


class EuroSATDataset(Dataset):
    def __init__(self, split, config, path=H5_PATH):
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {split}")
        self.path = Path(path).resolve()
        self.transform = model_transform(config, split == "train")
        
        with h5py.File(self.path, "r") as h5:
            self.indices = np.flatnonzero(h5["split"].asstr()[:] == split)
            self.labels = h5["labels"][:][self.indices]
            self.filenames = h5["filenames"].asstr()[:][self.indices]
            self.classes = h5["classes"].asstr()[:].tolist()
        
        self._file = None
        self._pid = None

    def __len__(self):
        return len(self.indices)

    def __getstate__(self):
        return dict(self.__dict__, _file=None, _pid=None)

    def __getitem__(self, index):
        
        if self._file is None or self._pid != os.getpid():
            self._file = h5py.File(self.path, "r")
            self._pid = os.getpid()
        
        tile = Image.fromarray(self._file["images"][self.indices[index]])
        
        return self.transform(tile), int(self.labels[index])


def make_loader(split, config, batch_size=64, num_workers=0, path=H5_PATH):
    dataset = EuroSATDataset(split, config, path)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=split == "train", num_workers=num_workers)
