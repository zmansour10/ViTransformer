"""Evaluate the validation-selected checkpoints on EuroSAT's held-out test images."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.dataset import make_loader
from station1.models import input_config
from station1.train import EXPERIMENTS, RESULTS, EuroSATClassifier, device_name, scores, write_csv


def evaluate(experiments, results, device, batch_size, workers):
    summary_path = results / "results.csv"
    summary = pd.read_csv(summary_path).set_index("experiment", verify_integrity=True)
    missing = set(experiments) - set(summary.index)
    
    if missing:
        raise ValueError(f"Training has not finished for {sorted(missing)}")

    records = []
    for name in experiments:
        path = results / name / "best.pt"
        classifier = EuroSATClassifier.load_from_checkpoint(path, map_location="cpu", pretrained=False)
    
        if classifier.hparams.settings["experiment"] != name:
            raise ValueError(f"{name}: checkpoint has a different experiment name")
    
        classes = classifier.hparams.classes
        settings = classifier.hparams.settings
        config = settings.get("input_config") or input_config(settings["architecture"], center_crop=True)
        loader = make_loader("test", config, batch_size, workers)
    
        if loader.dataset.classes != classes:
            raise ValueError(f"{name}: dataset class order differs from the checkpoint")
    
        trainer = pl.Trainer(accelerator=device, devices=1, logger=False, enable_checkpointing=False,
                             enable_model_summary=False, enable_progress_bar=False)
    
        batches = trainer.predict(classifier, dataloaders=loader)
        labels, probs, losses = zip(*batches)
        target = torch.cat(labels).cpu().numpy()
        probabilities = torch.cat(probs).cpu().numpy()
        loss = torch.cat(losses).mean().item()
        np.testing.assert_array_equal(target, loader.dataset.labels)
        predicted = probabilities.argmax(axis=1)
        matrix, precision, recall, f1 = scores(target, predicted, len(classes))
        summary.loc[name, ["test_loss", "test_accuracy", "test_macro_f1"]] = (
            loss, float((target == predicted).mean()), float(f1.mean()))
        per_class = {
            label: dict(support=int(matrix[i].sum()), precision=precision[i],
                        recall=recall[i], f1=f1[i])
            for i, label in enumerate(classes)
        }
    
        # Rows are true classes and columns are predicted classes in the recorded order.
        summary.at[name, "test_class_order"] = json.dumps(classes)
        summary.at[name, "test_confusion_matrix"] = json.dumps(matrix.tolist())
        summary.at[name, "test_per_class"] = json.dumps(per_class)
    
        records.append(pd.DataFrame(dict(experiment=name, split="test",
                                         filename=loader.dataset.filenames,
                                         true_label=target, true_class=np.array(classes)[target],
                                         predicted_label=predicted, predicted_class=np.array(classes)[predicted],
                                         confidence=probabilities.max(axis=1))))
    
        print(f"{name}  test accuracy {(target == predicted).mean():.4f} test macro-F1 {f1.mean():.4f}", flush=True)

    predictions_path = results / "predictions.csv"
    
    if predictions_path.exists():
        previous = pd.read_csv(predictions_path)
        records.insert(0, previous[~previous["experiment"].isin(experiments)])
    
    write_csv(pd.concat(records, ignore_index=True), predictions_path)
    write_csv(summary.reset_index(), summary_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=EXPERIMENTS)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default=device_name())
    
    args = parser.parse_args()
    
    evaluate(list(dict.fromkeys(args.experiments)), args.results, args.device,
             args.batch_size, args.workers)
