# EuroSAT — Station 1

A small research project comparing **pretrained CNN transfer learning**, **ViT linear probing**, and **ViT + LoRA** on EuroSAT RGB.

## Work in this order

1. **`eda.ipynb`** — Existing dataset exploration, including the fixed image split.
2. **`prepare_data.py`** — Convert the original images and existing split into `data/eurosat.h5`.
3. **`dataset.py`** — Read the HDF5 dataset and apply the transforms required by each pretrained checkpoint.
4. **`models.py`** — Define a pretrained CNN, frozen ViT with classification head, and ViT with LoRA (ranks 4, 8, 16).
5. **`train.py`** — Run CNN, ViT linear, then the three LoRA configurations, saving the best weights and training history.
6. **`evaluate.py`** — Evaluate on the untouched test set and save per-image predictions.
7. **`make_figures.py`** — Create the final figures directly from the saved results.
8. **`paper/`** — Put the finished six-page CVPR-style PDF here.

The two original notebooks are included unchanged. The six `.py` files are **intentionally not implemented**; write the coursework code yourself, following the lecturer's source-code rules. `preprocessing.ipynb` is kept as a reference, so its data-preparation logic can be reviewed before implementing the HDF5 step.

## Data and outputs

- Put the original EuroSAT class folders inside `data/EuroSAT_RGB/` (not included in this ZIP).
- **Keep the split you already generated** (e.g. `eurosat_split.csv`); do not silently resplit the dataset. Place a copy at `data/eurosat_split.csv` when available.
- `data/eurosat.h5` should contain original 64×64 RGB images, labels, filenames and split membership. Resize/normalize when loading for each model.
- Every training run saves `results/<run_name>/best.pt` and `results/<run_name>/history.csv`.
- Use **one** `results/results.csv` for configurations, validation/test metrics and efficiency measurements, and **one** `results/predictions.csv` for test predictions from all models.
- Save only the report-worthy plots in `results/figures/`.

The HDF5, result CSVs and model checkpoints are **not created yet**. They are the outputs of the scripts once you implement and run them. For reviewing results together, share the CSVs, histories and figures; usually there is no need to share the dataset or weights.

See `station1_script_checklist.md` for the specific TODOs and expected outputs.
