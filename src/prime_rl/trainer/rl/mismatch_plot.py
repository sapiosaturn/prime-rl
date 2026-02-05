from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    x64 = x.astype(np.float64, copy=False)
    y64 = y.astype(np.float64, copy=False)
    x64 = x64 - x64.mean()
    y64 = y64 - y64.mean()
    denom = float(np.sqrt(np.sum(x64 * x64) * np.sum(y64 * y64)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(x64 * y64) / denom)


def _downsample_pairs(
    trainer_probs: np.ndarray, inference_probs: np.ndarray, *, max_points: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if trainer_probs.shape != inference_probs.shape:
        raise ValueError(f"Shape mismatch: trainer={trainer_probs.shape}, inference={inference_probs.shape}")

    n = int(trainer_probs.size)
    if n <= max_points:
        return trainer_probs, inference_probs

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=int(max_points), replace=False)
    return trainer_probs[idx], inference_probs[idx]


def save_mismatch_plot(
    *,
    trainer_probs: Sequence[float],
    inference_probs: Sequence[float],
    out_path: Path,
    note: str | None = None,
    step: int,
    max_points: int,
) -> float:
    tp = np.asarray(trainer_probs, dtype=np.float32)
    ip = np.asarray(inference_probs, dtype=np.float32)
    tp, ip = _downsample_pairs(tp, ip, max_points=max_points, seed=step)

    abs_diff = np.abs(tp - ip)
    corr = _pearson_r(ip, tp)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=150)
    if note:
        fig.suptitle(note, fontsize=10)
    sc = ax.scatter(ip, tp, c=abs_diff, s=6, alpha=0.7)
    ax.plot([0.0, 1.0], [0.0, 1.0], "r--", linewidth=1.5, label="Perfect Precision (y=x)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Inference Probability")
    ax.set_ylabel("Training Probability")
    ax.set_title(f"Trainer–Inference Probability Mismatch (Step {step})")
    ax.legend(loc="upper left")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Absolute Probability Difference")

    ax.text(
        0.98,
        0.02,
        f"Correlation: {corr:.6f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )

    if note:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    else:
        fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return corr

