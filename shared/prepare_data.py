"""Pack the original EuroSAT tiles in the order of the fixed split CSV."""

import csv
from pathlib import Path
from tempfile import NamedTemporaryFile

import h5py
import numpy as np
from PIL import Image


DATA = Path(__file__).resolve().parents[1] / "data"


def prepare_data(image_dir=DATA / "EuroSAT_RGB", split_csv=DATA / "eurosat_split.csv",
                 output=DATA / "eurosat.h5"):
    image_dir, output = Path(image_dir), Path(output)
    
    with open(split_csv, newline="") as source:
        rows = list(csv.DictReader(source))

    files = [row["file"] for row in rows]
    
    originals = {p.relative_to(image_dir).as_posix() for p in image_dir.glob("*/*.jpg")}
    
    if len(files) != 27000 or len(set(files)) != len(files) or set(files) != originals:
        raise ValueError("The split CSV must list each of the 27,000 source tiles exactly once.")

    classes = sorted({Path(name).parent.name for name in files})
    labels = np.array([int(row["label"]) for row in rows])
    splits = [row["split"] for row in rows]
    
    if len(classes) != 10 or set(splits) != {"train", "validation", "test"}:
        raise ValueError("Expected the ten EuroSAT classes and train, validation and test splits.")
    
    for row, label in zip(rows, labels):
        if not 0 <= label < len(classes) or classes[label] != row["class"] or Path(row["file"]).parent.name != row["class"]:
            raise ValueError(f"Inconsistent class or label for {row['file']}")

    with NamedTemporaryFile(dir=output.parent, suffix=".h5", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        with h5py.File(temporary, "w") as h5:
            images = h5.create_dataset("images", (len(files), 64, 64, 3), dtype="uint8")
    
            for i, name in enumerate(files):
                with Image.open(image_dir / name) as tile:
    
                    if tile.mode != "RGB" or tile.size != (64, 64):
                        raise ValueError(f"Unexpected image format for {name}")
    
                    images[i] = np.asarray(tile)
            h5["labels"] = labels
    
            for key, values in [("filenames", files), ("split", splits), ("classes", classes)]:
                h5.create_dataset(key, data=values, dtype=h5py.string_dtype())
    
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


if __name__ == "__main__":
    print(prepare_data())
