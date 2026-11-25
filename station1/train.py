"""Train Station 1 experiments and select checkpoints by validation macro-F1."""

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from shared.dataset import make_loader
from station1.models import LORA_RANKS, VIT, build_model, input_config

RESULTS = ROOT / "station1" / "results"
EXPERIMENTS = ("cnn", "vit_linear", *LORA_RANKS)


def device_name():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def write_csv(frame, path):
    temp = path.with_suffix(".csv.tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def scores(labels, predictions, n_classes):
    matrix = np.bincount(labels * n_classes + predictions,
                         minlength=n_classes**2).reshape(n_classes, n_classes)
    
    hits = matrix.diagonal()
    support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    precision = hits / predicted.clip(min=1)
    recall = hits / support.clip(min=1)
    f1 = 2 * hits / (support + predicted).clip(min=1)
    
    return matrix, precision, recall, f1


class EuroSATClassifier(pl.LightningModule):
    def __init__(self, settings, classes, pretrained=True):
        super().__init__()
        self.save_hyperparameters(ignore=["pretrained"])
        self.model = build_model(settings["experiment"], len(classes), pretrained=pretrained,
                                 vit_arch=settings["architecture"])
        self.validation_labels = []
        self.validation_predictions = []

    def forward(self, images):
        return self.model(images)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = nn.functional.cross_entropy(logits, labels)
    
        self.log_dict({"train_loss": loss, "train_accuracy": (logits.argmax(1) == labels).float().mean()},
                      on_step=False, on_epoch=True, batch_size=len(labels))
    
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        self.log("val_loss", nn.functional.cross_entropy(logits, labels), batch_size=len(labels))
        self.validation_labels.append(labels.cpu())
        self.validation_predictions.append(logits.argmax(1).cpu())

    def on_validation_epoch_end(self):
        labels = torch.cat(self.validation_labels).numpy()
        preds = torch.cat(self.validation_predictions).numpy()
        f1 = scores(labels, preds, len(self.hparams.classes))[-1]
        self.log("val_macro_f1", float(f1.mean()))
        self.log("val_accuracy", float((labels == preds).mean()))
        self.validation_labels.clear()
        self.validation_predictions.clear()

    def predict_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
    
        return labels, logits.softmax(1), nn.functional.cross_entropy(logits, labels, reduction="none")

    def configure_optimizers(self):
        settings = self.hparams.settings
    
        return torch.optim.AdamW((p for p in self.parameters() if p.requires_grad),
                                 lr=settings["learning_rate"], weight_decay=settings["weight_decay"])


class BestCheckpoint(ModelCheckpoint):
    FILE_EXTENSION = ".pt"


class TrainingHistory(pl.Callback):
    def __init__(self, path):
        self.path = path
        self.rows = []

    def on_train_epoch_start(self, trainer, pl_module):
        self.start = perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        row = {"epoch": trainer.current_epoch + 1, "epoch_seconds": perf_counter() - self.start}
        
        for key in ("train_loss", "train_accuracy", "val_loss", "val_accuracy", "val_macro_f1"):
            row[key] = trainer.callback_metrics[key].item()
        
        self.rows.append(row)
        
        write_csv(pd.DataFrame(self.rows), self.path)
        
        print(f"{self.path.parent.name} epoch {row['epoch']} validation macro-F1 {row['val_macro_f1']:.4f}",
              flush=True)


def train(name, args):
    folder = args.results / name
    folder.mkdir(parents=True, exist_ok=True)
    best = folder / "best.pt"
    
    if best.exists() or (folder / "history.csv").exists():
        raise FileExistsError(f"{folder} contains a run. Use --results for a separate run.")
    
    architecture = "resnet18" if name == "cnn" else VIT
    
    config = input_config(architecture)
    
    train_loader = make_loader("train", config, args.batch_size, args.workers)
    validation = make_loader("validation", config, args.batch_size, args.workers)
    
    dataset = train_loader.dataset
    classes = dataset.classes
    
    if args.train_per_class is not None:
        rng = np.random.default_rng(args.seed)
        indices = np.concatenate([
            rng.choice(np.flatnonzero(dataset.labels == label), args.train_per_class, replace=False)
            for label in range(len(classes))
        ])
        train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                                  sampler=SubsetRandomSampler(indices))
    pl.seed_everything(args.seed, workers=True)
    
    settings = dict(experiment=name, architecture=architecture, input_config=config,
                    weights="IMAGENET1K_V1" if name == "cnn" else VIT,
                    rank=LORA_RANKS.get(name, 0), alpha=LORA_RANKS.get(name, 0),
                    optimizer="AdamW", learning_rate=args.lr or (0.0003 if name in LORA_RANKS else 0.001),
                    patience=args.patience, weight_decay=0.01,
                    batch_size=args.batch_size, epochs=args.epochs, seed=args.seed,
                    training_images=len(train_loader.sampler),
                    workers=args.workers, device=args.device, torch_version=str(torch.__version__),
                    preprocessing=str(validation.dataset.transform),
                    augmentation="horizontal and vertical flips, p=0.5 each",
                    lightning_version=pl.__version__)
    
    classifier = EuroSATClassifier(settings, classes)
    
    settings["trainable_params"] = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    
    settings["total_params"] = sum(p.numel() for p in classifier.parameters())
    
    history = TrainingHistory(folder / "history.csv")
    
    checkpoint = BestCheckpoint(dirpath=folder, filename="best", monitor="val_macro_f1",
                                mode="max", save_top_k=1, save_weights_only=True,
                                enable_version_counter=False)
    
    stopping = EarlyStopping(monitor="val_macro_f1", mode="max", patience=args.patience)
    
    
    trainer = pl.Trainer(accelerator=args.device, devices=1, max_epochs=args.epochs,
                         callbacks=[history, checkpoint, stopping], logger=False, enable_progress_bar=False,
                         enable_model_summary=False, num_sanity_val_steps=0)
    
    
    pl.seed_everything(args.seed, workers=True)
    start = perf_counter()
    
    trainer.fit(classifier, train_dataloaders=train_loader, val_dataloaders=validation)
    
    elapsed = perf_counter() - start
    
    selected = max(history.rows, key=lambda row: row["val_macro_f1"])
    
    summary = dict(**settings, training_seconds=elapsed, best_epoch=selected["epoch"],
                   val_accuracy=selected["val_accuracy"], val_macro_f1=selected["val_macro_f1"],
                   val_loss=selected["val_loss"])
    
    path = args.results / "results.csv"
    
    previous = pd.read_csv(path) if path.exists() else pd.DataFrame()
    
    write_csv(pd.concat([previous, pd.DataFrame([summary])], ignore_index=True), path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=EXPERIMENTS)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, help="Override the head-only or LoRA learning rate")
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--train-per-class", type=int, help="Seeded training subset for an LR pilot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=device_name())
    
    args = parser.parse_args()
    
    if min(args.epochs, args.batch_size, args.patience) < 1 or args.workers < 0:
        parser.error("epochs, batch size and patience must be positive; workers must be nonnegative")
    
    if args.lr is not None and args.lr <= 0:
        parser.error("learning rate must be positive")
    
    if args.train_per_class is not None and (args.train_per_class < 1 or args.results.resolve() == RESULTS):
        parser.error("a subset pilot requires a positive --train-per-class and a separate --results folder")
    
    for experiment in args.experiments:
        train(experiment, args)
