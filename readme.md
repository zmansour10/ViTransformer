# Deep Learning   for Computer Vision

Course project for Deep Learning at HSBI.

# Station 1

Comparing pretrained CNN transfer learning, ViT linear probing and ViT + LoRA on EuroSAT RGB.

## Dataset 
### EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification

- Published:  July 22, 2018 | Version v2
- Authors/Creators: Helber, Patric; Bischke, Benjamin; Dengel, Andreas; Borth, Damian

![EuroSAT Overview](https://github.com/phelber/EuroSAT/blob/master/eurosat_overview_small.jpg)

- **Description:** EuroSAT is a land use and land cover classification dataset. The dataset is based on Sentinel-2 satellite imagery covering 13 spectral bands and consists of 10 LULC classes with a total of 27,000 labeled and geo-referenced images. The dataset is associated with the publications "Introducing EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification" and "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification".

- **EuroSAT_RGB.zip** contains the RGB version of the dataset, which includes the optical R, G and B frequency bands encoded as JPEG images.

- **EuroSAT_MS.zip** contains the multi-spectral version of the EuroSAT dataset, which includes all 13 Sentinel-2 bands in the original value range.

- The EuroSat official GitHub repo: https://github.com/phelber/EuroSAT

- The RGB dataset can be found here: https://zenodo.org/records/7711810#.ZAm3k-zMKEA

## Running the project

The project uses Poetry and was tested with Python 3.15.2.

The second command packs the images from `data/EuroSAT_RGB/` using the fixed split in `data/eurosat_split.csv`. You can skip it if `data/eurosat.h5` already exists.

```bash
poetry install
poetry run python shared/prepare_data.py
```

### Plot the saved results

This reads the saved runs in `station1/results/` and writes plots and classification reports to `station1/results/figures/`. Omit `--detailed` for just the four overview plots.


```bash
poetry run python shared/make_figures.py --detailed
```


### Train and evaluate

To train all five models with PyTorch Lightning, use a fresh results folder:

```bash
poetry run python station1/train.py --results station1/runs/my_run
poetry run python station1/evaluate.py --results station1/runs/my_run
poetry run python shared/make_figures.py --results station1/runs/my_run --detailed
```

Each model gets a `history.csv` and a `best.pt` checkpoint selected by validation macro-F1. Evaluation writes the test metrics to `results.csv` and the per-image predictions to `predictions.csv`. To run a single model, add the same option to both the training and evaluation commands, for example `--experiments vit_lora_r8`.

