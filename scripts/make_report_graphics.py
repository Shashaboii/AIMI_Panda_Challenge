"""Generate report-ready graphics from the logged PANDA results.

The figures are written to docs/graphics/ as both PNG and PDF where possible.
They are intentionally driven by local result CSVs and the numbers in
results.md so the report can cite the same evidence as the experiment log.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.metrics import confusion_matrix


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "graphics"
TILE_DIR = REPO_ROOT / "panda_tiles_36x192_png"

sys.path.insert(0, str(REPO_ROOT))
from src.eval import qwk, round_preds  # noqa: E402


COLORS = {
    "smooth": "#4C78A8",
    "ordinal": "#54A24B",
    "mse": "#A05D56",
    "tile": "#E45756",
    "gray": "#6E7781",
    "light_gray": "#E8ECEF",
    "dark": "#222222",
}


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(
            OUT_DIR / f"{stem}.{ext}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def load_oof() -> tuple[pd.DataFrame, pd.DataFrame]:
    smooth = pd.read_csv(REPO_ROOT / "5fold_smoothl1_oof_predictions.csv")
    ordinal = pd.read_csv(REPO_ROOT / "5fold_ordinal_oof_predictions.csv")
    return smooth, ordinal


def fold_scores(df: pd.DataFrame, pred_col: str) -> list[float]:
    scores = []
    for fold in sorted(df["fold"].unique()):
        part = df[df["fold"] == fold]
        scores.append(qwk(part[pred_col].values, part["isup_grade"].values))
    return scores


def make_oof_model_comparison(smooth: pd.DataFrame, ordinal: pd.DataFrame) -> None:
    values = [
        ("B0 thumbnail + MSE", 0.7030, COLORS["mse"]),
        ("B0 thumbnail + SmoothL1", qwk(smooth["pred_b0_smoothl1"], smooth["isup_grade"]), COLORS["smooth"]),
        ("B0 thumbnail + ordinal BCE", qwk(ordinal["pred_b0_ordinal"], ordinal["isup_grade"]), COLORS["ordinal"]),
        ("B0 thumbnail + ordinal BCE\n(clean 5-epoch rerun)", 0.7221, "#88C35F"),
    ]

    labels = [v[0] for v in values]
    scores = np.array([v[1] for v in values])
    colors = [v[2] for v in values]
    y = np.arange(len(values))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.barh(y, scores, color=colors, height=0.62)
    ax.axvline(0.83, color=COLORS["dark"], linestyle="--", linewidth=1.2)
    ax.text(0.832, len(values) - 0.55, "project target 0.83", va="center", fontsize=8)

    for bar, score in zip(bars, scores):
        ax.text(
            score + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}",
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.set_yticks(y, labels)
    ax.set_xlim(0.68, 0.85)
    ax.set_xlabel("Global OOF QWK")
    ax.set_title("Thumbnail OOF Baselines")
    ax.grid(axis="x", color=COLORS["light_gray"], linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save_figure(fig, "qwk_oof_model_comparison")


def make_fold_qwk_comparison(smooth: pd.DataFrame, ordinal: pd.DataFrame) -> None:
    smooth_scores = np.array(fold_scores(smooth, "pred_b0_smoothl1"))
    ordinal_scores = np.array(fold_scores(ordinal, "pred_b0_ordinal"))
    folds = np.arange(5)
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(folds - width / 2, smooth_scores, width, label="SmoothL1", color=COLORS["smooth"])
    ax.bar(folds + width / 2, ordinal_scores, width, label="Ordinal BCE", color=COLORS["ordinal"])

    for x, score in zip(folds - width / 2, smooth_scores):
        ax.text(x, score + 0.002, f"{score:.3f}", ha="center", va="bottom", fontsize=8)
    for x, score in zip(folds + width / 2, ordinal_scores):
        ax.text(x, score + 0.002, f"{score:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(folds, [f"fold {i}" for i in folds])
    ax.set_ylim(0.68, 0.75)
    ax.set_ylabel("QWK")
    ax.set_title("Ordinal BCE Beats SmoothL1 on Every Fold")
    ax.legend(frameon=False, ncols=2, loc="upper left")
    ax.grid(axis="y", color=COLORS["light_gray"], linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, "fold_qwk_smoothl1_vs_ordinal")


def make_recall_comparison(smooth: pd.DataFrame, ordinal: pd.DataFrame) -> None:
    labels = np.arange(6)
    smooth_pred = round_preds(smooth["pred_b0_smoothl1"].values)
    ordinal_pred = round_preds(ordinal["pred_b0_ordinal"].values)
    y_true = smooth["isup_grade"].values

    cm_smooth = confusion_matrix(y_true, smooth_pred, labels=labels)
    cm_ordinal = confusion_matrix(y_true, ordinal_pred, labels=labels)
    rec_smooth = np.diag(cm_smooth) / cm_smooth.sum(axis=1)
    rec_ordinal = np.diag(cm_ordinal) / cm_ordinal.sum(axis=1)

    x = np.arange(6)
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - width / 2, rec_smooth, width, label="SmoothL1", color=COLORS["smooth"])
    ax.bar(x + width / 2, rec_ordinal, width, label="Ordinal BCE", color=COLORS["ordinal"])

    for xpos, vals in ((x - width / 2, rec_smooth), (x + width / 2, rec_ordinal)):
        for xx, val in zip(xpos, vals):
            ax.text(xx, val + 0.015, f"{val * 100:.0f}%", ha="center", fontsize=8)

    ax.set_xticks(x, [str(i) for i in labels])
    ax.set_ylim(0, 0.82)
    ax.set_xlabel("True ISUP grade")
    ax.set_ylabel("Recall")
    ax.set_title("Per-grade Recall from Recovered OOF Predictions")
    ax.legend(frameon=False, ncols=2, loc="upper right")
    ax.grid(axis="y", color=COLORS["light_gray"], linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, "isup_recall_smoothl1_vs_ordinal")


def make_confusion_pair(smooth: pd.DataFrame, ordinal: pd.DataFrame) -> None:
    labels = np.arange(6)
    configs = [
        ("SmoothL1 recovered OOF\nQWK 0.7147", smooth, "pred_b0_smoothl1"),
        ("Ordinal BCE recovered OOF\nQWK 0.7244", ordinal, "pred_b0_ordinal"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8), constrained_layout=True)
    for ax, (title, df, pred_col) in zip(axes, configs):
        preds = round_preds(df[pred_col].values)
        cm = confusion_matrix(df["isup_grade"].values, preds, labels=labels)
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        image = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=0.75)
        ax.set_title(title)
        ax.set_xlabel("Predicted ISUP")
        ax.set_ylabel("True ISUP")
        ax.set_xticks(labels)
        ax.set_yticks(labels)
        for i in labels:
            for j in labels:
                color = "white" if cm_norm[i, j] > 0.42 else COLORS["dark"]
                ax.text(
                    j,
                    i,
                    f"{cm_norm[i, j] * 100:.0f}%\n{cm[i, j]}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=color,
                )
    cbar = fig.colorbar(image, ax=axes, fraction=0.045, pad=0.02)
    cbar.set_label("Row-normalized percentage")
    save_figure(fig, "confusion_matrices_smoothl1_vs_ordinal")


def make_tile_artifact_audit() -> None:
    real_mean = 5.356
    blank_mean = 30.644
    real_median = 5.0
    max_real = 9

    tile_scores = [
        ("Thumbnail\nordinal fold 0", 0.7314, COLORS["ordinal"]),
        ("Invalid tiles\nN=16", 0.6819, COLORS["tile"]),
        ("Invalid tiles\nN=32", 0.6676, COLORS["tile"]),
        ("Invalid tiles\nN=36", 0.6720, COLORS["tile"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), gridspec_kw={"width_ratios": [0.9, 1.55]})

    ax = axes[0]
    ax.bar([0], [real_mean], color=COLORS["ordinal"], width=0.56, label="real tiles")
    ax.bar([0], [blank_mean], bottom=[real_mean], color=COLORS["light_gray"], edgecolor="#BBC3CA", width=0.56, label="blank padding")
    ax.axhline(36, color=COLORS["dark"], linewidth=0.9)
    ax.text(0.33, real_mean / 2, f"mean real\n{real_mean:.1f}", va="center", fontsize=9)
    ax.text(0.33, real_mean + blank_mean / 2, f"mean blank\n{blank_mean:.1f}", va="center", fontsize=9)
    ax.text(-0.42, 37.2, f"36 requested tiles\nmedian real={real_median:.0f}, max real={max_real}", fontsize=8)
    ax.set_xlim(-0.65, 0.9)
    ax.set_ylim(0, 40)
    ax.set_xticks([])
    ax.set_ylabel("Tile slots per slide")
    ax.set_title("Artifact Audit")
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    ax.spines[["top", "right", "bottom"]].set_visible(False)

    ax = axes[1]
    x = np.arange(len(tile_scores))
    scores = [item[1] for item in tile_scores]
    colors = [item[2] for item in tile_scores]
    ax.bar(x, scores, color=colors, width=0.62)
    for xx, score in zip(x, scores):
        ax.text(xx, score + 0.006, f"{score:.4f}", ha="center", fontsize=8)
    ax.set_xticks(x, [item[0] for item in tile_scores])
    ax.set_ylim(0.64, 0.75)
    ax.set_ylabel("Fold-0 QWK")
    ax.set_title("Tile Sweep Was a Bad-input Result")
    ax.grid(axis="y", color=COLORS["light_gray"], linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Current Tile Artifacts Are Mostly White Padding", y=1.02, fontsize=13)
    save_figure(fig, "tile_artifact_audit")


def count_estimated_real_tiles(path: Path, tile_size: int = 192, threshold: float = 250.0) -> int:
    arr = np.asarray(Image.open(path))
    tiles = arr.reshape(36, tile_size, tile_size, 3)
    return int((tiles.mean(axis=(1, 2, 3)) < threshold).sum())


def make_invalid_tile_example() -> None:
    if not TILE_DIR.exists():
        return

    candidates = list(TILE_DIR.glob("*.png"))[:100]
    if not candidates:
        return

    chosen = max(candidates, key=count_estimated_real_tiles)
    im = Image.open(chosen).convert("RGB")
    tile_size = 192
    tiles = [im.crop((0, i * tile_size, tile_size, (i + 1) * tile_size)) for i in range(36)]
    real_flags = [np.asarray(tile).mean() < 250.0 for tile in tiles]

    grid = Image.new("RGB", (6 * tile_size, 6 * tile_size), "white")
    for idx, tile in enumerate(tiles):
        row, col = divmod(idx, 6)
        grid.paste(tile, (col * tile_size, row * tile_size))

    draw = ImageDraw.Draw(grid)
    for idx, is_real in enumerate(real_flags):
        row, col = divmod(idx, 6)
        x0 = col * tile_size
        y0 = row * tile_size
        x1 = x0 + tile_size - 1
        y1 = y0 + tile_size - 1
        color = "#2F855A" if is_real else "#A0A7AF"
        width = 5 if is_real else 2
        draw.rectangle((x0, y0, x1, y1), outline=color, width=width)

    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    ax.imshow(grid)
    ax.set_axis_off()
    ax.set_title(
        f"Example 36-tile artifact: {sum(real_flags)} estimated tissue tiles, {36 - sum(real_flags)} blank/pale slots",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.02,
        f"Source artifact: {chosen.name}. Green outlines mark estimated tissue-rich tiles.",
        ha="center",
        fontsize=8,
        color=COLORS["gray"],
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "invalid_tile_artifact_example.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_summary_csv(smooth: pd.DataFrame, ordinal: pd.DataFrame) -> None:
    rows = [
        {"figure": "qwk_oof_model_comparison", "metric": "MSE 5-fold OOF QWK", "value": 0.7030},
        {
            "figure": "qwk_oof_model_comparison",
            "metric": "SmoothL1 recovered OOF QWK",
            "value": qwk(smooth["pred_b0_smoothl1"], smooth["isup_grade"]),
        },
        {
            "figure": "qwk_oof_model_comparison",
            "metric": "Ordinal recovered OOF QWK",
            "value": qwk(ordinal["pred_b0_ordinal"], ordinal["isup_grade"]),
        },
        {"figure": "qwk_oof_model_comparison", "metric": "Ordinal clean 5-epoch OOF QWK", "value": 0.7221},
        {"figure": "tile_artifact_audit", "metric": "mean real tiles", "value": 5.356},
        {"figure": "tile_artifact_audit", "metric": "mean blank tiles", "value": 30.644},
        {"figure": "tile_artifact_audit", "metric": "maximum audited real tiles", "value": 9},
    ]
    pd.DataFrame(rows).to_csv(OUT_DIR / "report_graphics_summary.csv", index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    smooth, ordinal = load_oof()
    make_oof_model_comparison(smooth, ordinal)
    make_fold_qwk_comparison(smooth, ordinal)
    make_recall_comparison(smooth, ordinal)
    make_confusion_pair(smooth, ordinal)
    make_tile_artifact_audit()
    make_invalid_tile_example()
    write_summary_csv(smooth, ordinal)

    for path in sorted(OUT_DIR.iterdir()):
        print(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
