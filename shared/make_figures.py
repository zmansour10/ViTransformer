"""Plot a station's saved metrics, histories and predictions without model inference."""

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NAMES = {"cnn": "ResNet18", "vit_linear": "ViT linear probe"}
COLORS = ["#0072B2", "#D55E00", "#009E73", "#8C5694", "#986500"]


def model_name(name):
    if name.startswith("vit_lora_r"):
        return "LoRA r=" + name.removeprefix("vit_lora_r")
    
    return NAMES.get(name, name.replace("_", " "))


def comparison(results):
    fig, ax = plt.subplots(figsize=(6.8, 3.3), layout="constrained")
    positions = np.arange(len(results))
    
    for column, label, color, marker, offset in [
        ("test_accuracy", "Accuracy", COLORS[0], "o", -0.13),
        ("test_macro_f1", "Macro-F1", COLORS[1], "s", 0.13),
    ]:
        ax.scatter(results[column] * 100, positions + offset, marker=marker,
                   color=color, label=label, s=35)
    ax.set_yticks(positions, [model_name(name) for name in results.index])
    ax.invert_yaxis()
    ax.set_xlabel("Held-out test score (%)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=2)
    ax.grid(axis="x")
    
    return fig


def lora_efficiency(runs):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2), layout="constrained", sharey=True)
    
    for ax, column, scale, title in [
        (axes[0], "trainable_params", 1000, "Trainable parameters (thousands)"),
        (axes[1], "training_seconds", 3600, "Training and validation time (h)"),
    ]:
        x = runs[column] / scale
        y = runs.test_macro_f1 * 100
        ax.scatter(x, y, color=COLORS[0])
    
        for row, xval, yval in zip(runs.itertuples(), x, y):
            ax.annotate(f"r={int(row.rank)}", (xval, yval), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=8)
    
        ax.margins(x=0.18, y=0.35)
        ax.grid(axis="y")
        ax.set_xlabel(title)
    axes[0].set_ylabel("Test macro-F1 (%)")
    
    return fig


def training_curves(results, directory):
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.6), layout="constrained")
    handles = []
    
    for i, (name, run) in enumerate(results.iterrows()):
        
        history = pd.read_csv(directory / name / "history.csv").set_index("epoch")
        color = COLORS[i % len(COLORS)]
        line, = axes[0].plot(history.index, history.train_loss, color=color, label=model_name(name))
        handles.append(line)
        axes[1].plot(history.index, history.val_macro_f1 * 100, color=color)
        epoch = int(run.best_epoch)
        axes[1].scatter(epoch, history.loc[epoch, "val_macro_f1"] * 100,
                        s=25, color=color, edgecolor="black", linewidth=0.5, zorder=3)
    
    for ax, ylabel in zip(axes, ["Training cross-entropy", "Validation macro-F1 (%)"]):
    
        ax.set(xlabel="Epoch", ylabel=ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y")
    
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    
    return fig


def error_examples(predictions, model, data):
    errors = predictions.loc[(predictions.experiment == model) & (predictions.split == "test")
                             & (predictions.true_label != predictions.predicted_label)]
    pairs = errors.groupby(["true_class", "predicted_class"], sort=True)
    frequent = pairs.size().sort_values(ascending=False, kind="stable").head(4)
    fig, axes = plt.subplots(1, max(len(frequent), 1), figsize=(7.2, 2.7),
                             squeeze=False, layout="constrained")
    fig.suptitle(f"Test errors from {model_name(model)}")
    
    if frequent.empty:
        axes[0, 0].text(0.5, 0.5, "All test images classified correctly", ha="center")
        axes[0, 0].set_axis_off()
        return fig
    
    with h5py.File(data, "r") as h5:
        indices = {name: i for i, name in enumerate(h5["filenames"].asstr()[:])}
    
        for ax, (pair, count) in zip(axes.flat, frequent.items()):
            group = pairs.get_group(pair).sort_values("filename")
            typical = group.loc[(group.confidence - group.confidence.median()).abs().idxmin()]
            ax.imshow(h5["images"][indices[typical.filename]], interpolation="nearest")
            ax.set(xticks=[], yticks=[], title=f"True: {pair[0]}\nPred: {pair[1]}",
                   xlabel=f"{count} such errors\nConfidence {typical.confidence:.0%}")
            ax.title.set_fontsize(8)
    
    return fig


def make_figures(directory, data):
    results = pd.read_csv(directory / "results.csv").set_index("experiment", verify_integrity=True)
    predictions = pd.read_csv(directory / "predictions.csv")
    
    if results[["test_accuracy", "test_macro_f1"]].isna().any().any():
        raise ValueError("Complete held-out evaluation for each run before plotting.")
    
    folder = directory / "figures"
    folder.mkdir(exist_ok=True)

    def save(fig, name):
        fig.savefig(folder / (name + ".pdf"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    save(comparison(results), "model_comparison")
    
    if "rank" in results:
        runs = results.loc[results["rank"] > 0].sort_values("rank")
    
        if len(runs):
            save(lora_efficiency(runs), "lora_efficiency")
    
    save(training_curves(results, directory), "training_curves")
    model = results.val_macro_f1.idxmax()
    
    if not ((predictions.experiment == model) & (predictions.split == "test")).any():
        raise ValueError(f"Missing test predictions for {model}")
    
    save(error_examples(predictions, model, data), "error_examples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument("--results", type=Path, default=ROOT / "station1" / "results")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "eurosat.h5")
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.axisbelow": True, "grid.alpha": 0.25, "legend.frameon": False,
                         "pdf.fonttype": 42})
    
    make_figures(args.results, args.data)
