#!/usr/bin/env python3
"""Visualise training metrics from JSONL log files produced by MetricsLogger.

Usage:
    python scripts/plot_metrics.py checkpoints/metrics.jsonl
    python scripts/plot_metrics.py checkpoints/metrics.jsonl --out figures/
    python scripts/plot_metrics.py run1/metrics.jsonl run2/metrics.jsonl --labels "run1" "run2"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.figsize": (10, 6),
})


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def smooth(values: List[float], window: int = 5) -> List[float]:
    if window <= 1 or len(values) <= window:
        return values
    out: List[float] = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo : i + 1]) / (i - lo + 1))
    return out


def _get(records: List[dict], key: str) -> tuple:
    steps = [r["step"] for r in records if key in r]
    vals = [r[key] for r in records if key in r]
    return steps, vals


def detect_trainer_type(records: List[dict]) -> str:
    keys = set()
    for r in records:
        keys.update(r.keys())
    if "challenger_loss" in keys or "challenger_reward" in keys:
        return "adversarial"
    if "policy_loss" in keys or "kl_penalty" in keys:
        return "grpo"
    return "reinforce"


def plot_adversarial(records: List[dict], out_dir: Path, label: str = "",
                     window: int = 5) -> List[Path]:
    saved: List[Path] = []
    prefix = f"{label}_" if label else ""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Adversarial Training{' — ' + label if label else ''}", fontsize=15)

    # (0,0) J_score
    ax = axes[0, 0]
    s, v = _get(records, "j_score")
    if v:
        ax.plot(s, smooth(v, window), alpha=0.4, label="per-step")
    s2, v2 = _get(records, "avg_j_score")
    if v2:
        ax.plot(s2, v2, linewidth=2, label="running avg")
    ax.set_title("J_score (Solver)")
    ax.set_xlabel("step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (0,1) Rewards
    ax = axes[0, 1]
    for key, lbl, color in [
        ("solver_reward", "Solver Reward", "#2196F3"),
        ("challenger_reward", "Challenger Reward", "#FF5722"),
    ]:
        s, v = _get(records, key)
        if v:
            ax.plot(s, smooth(v, window), label=lbl, color=color)
    s2, v2 = _get(records, "avg_solver_reward")
    if v2:
        ax.plot(s2, v2, "--", linewidth=1.5, color="#1565C0", label="Solver avg")
    s3, v3 = _get(records, "avg_challenger_reward")
    if v3:
        ax.plot(s3, v3, "--", linewidth=1.5, color="#BF360C", label="Challenger avg")
    ax.set_title("Rewards")
    ax.set_xlabel("step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,0) Losses
    ax = axes[1, 0]
    for key, lbl in [
        ("solver_loss", "Solver Total Loss"),
        ("challenger_loss", "Challenger Total Loss"),
        ("solver_policy_loss", "Solver Policy Loss"),
        ("challenger_policy_loss", "Challenger Policy Loss"),
    ]:
        s, v = _get(records, key)
        if v:
            ax.plot(s, smooth(v, window), label=lbl)
    ax.set_title("Losses")
    ax.set_xlabel("step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,1) KL
    ax = axes[1, 1]
    for key, lbl in [
        ("solver_kl", "Solver KL"),
        ("challenger_kl", "Challenger KL"),
    ]:
        s, v = _get(records, key)
        if v:
            ax.plot(s, smooth(v, window), label=lbl)
    ax.set_title("KL Divergence")
    ax.set_xlabel("step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = out_dir / f"{prefix}adversarial_training.png"
    fig.savefig(p)
    plt.close(fig)
    saved.append(p)
    print(f"  Saved: {p}")
    return saved


def plot_grpo(records: List[dict], out_dir: Path, label: str = "",
              window: int = 5) -> List[Path]:
    saved: List[Path] = []
    prefix = f"{label}_" if label else ""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"GRPO Solver Training{' — ' + label if label else ''}", fontsize=15)

    ax = axes[0, 0]
    for key, lbl in [("correctness", "per-step"), ("avg_correctness", "running avg")]:
        s, v = _get(records, key)
        if v:
            ax.plot(s, smooth(v, window) if "avg" not in key else v, label=lbl)
    ax.set_title("Correctness (J_score)")
    ax.set_xlabel("step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for key, lbl in [("mean_reward", "per-step"), ("avg_reward", "running avg")]:
        s, v = _get(records, key)
        if v:
            ax.plot(s, smooth(v, window) if "avg" not in key else v, label=lbl)
    ax.set_title("Reward")
    ax.set_xlabel("step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for key, lbl in [
        ("total_loss", "Total Loss"),
        ("policy_loss", "Policy Loss"),
        ("avg_loss", "Avg Total Loss"),
    ]:
        s, v = _get(records, key)
        if v:
            ax.plot(s, smooth(v, window) if "avg" not in key else v, label=lbl)
    ax.set_title("Loss")
    ax.set_xlabel("step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    s, v = _get(records, "kl_penalty")
    if v:
        ax.plot(s, smooth(v, window), label="KL penalty")
    ax.set_title("KL Divergence")
    ax.set_xlabel("step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = out_dir / f"{prefix}grpo_training.png"
    fig.savefig(p)
    plt.close(fig)
    saved.append(p)
    print(f"  Saved: {p}")
    return saved


def plot_reinforce(records: List[dict], out_dir: Path, label: str = "",
                   window: int = 5) -> List[Path]:
    saved: List[Path] = []
    prefix = f"{label}_" if label else ""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Reinforce Pipeline{' — ' + label if label else ''}", fontsize=15)

    ax = axes[0, 0]
    for key, lbl in [("j_score", "per-step"), ("avg_j_score", "running avg")]:
        s, v = _get(records, key)
        if v:
            ax.plot(s, smooth(v, window) if "avg" not in key else v, label=lbl)
    ax.set_title("J_score")
    ax.set_xlabel("step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for key, lbl, color in [
        ("solver_reward", "Solver", "#2196F3"),
        ("challenge_reward", "Challenger", "#FF5722"),
    ]:
        s, v = _get(records, key)
        if v:
            ax.plot(s, smooth(v, window), label=lbl, color=color)
    ax.set_title("Rewards")
    ax.set_xlabel("step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for key, lbl in [
        ("r_adv", "R_adv"),
        ("r_rep", "R_rep"),
        ("r_fmt", "R_fmt"),
        ("r_rel", "R_rel"),
        ("r_rubric", "R_rubric"),
    ]:
        s, v = _get(records, key)
        if v:
            ax.plot(s, smooth(v, window), label=lbl)
    ax.set_title("Challenger Reward Components")
    ax.set_xlabel("step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    s, v = _get(records, "avg_solver_reward")
    if v:
        ax.plot(s, v, label="Avg Solver Reward", color="#1565C0")
    s2, v2 = _get(records, "avg_challenge_reward")
    if v2:
        ax.plot(s2, v2, label="Avg Challenger Reward", color="#BF360C")
    ax.set_title("Running Averages")
    ax.set_xlabel("step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = out_dir / f"{prefix}reinforce_training.png"
    fig.savefig(p)
    plt.close(fig)
    saved.append(p)
    print(f"  Saved: {p}")
    return saved


def plot_comparison(all_records: List[List[dict]], labels: List[str],
                    out_dir: Path, window: int = 5) -> List[Path]:
    """Compare multiple runs on a single figure (avg_j_score vs step)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors

    for i, (records, lbl) in enumerate(zip(all_records, labels)):
        for key in ("avg_j_score", "avg_correctness"):
            s, v = _get(records, key)
            if v:
                ax.plot(s, v, label=f"{lbl} ({key})", color=colors[i % len(colors)])
                break

    ax.set_title("Run Comparison — J_score / Correctness")
    ax.set_xlabel("step")
    ax.set_ylabel("score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_dir / "comparison.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  Saved: {p}")
    return [p]


def plot_final_bar(metrics_json: str | Path, out_dir: Path) -> List[Path]:
    """Bar chart from final_metrics.json produced by train_adversarial.py."""
    data = json.loads(Path(metrics_json).read_text())
    numeric = {k: v for k, v in data.items() if isinstance(v, (int, float))}
    if not numeric:
        return []

    fig, ax = plt.subplots(figsize=(max(8, len(numeric) * 0.6), 5))
    keys = list(numeric.keys())
    vals = list(numeric.values())
    bars = ax.barh(keys, vals, color="#42A5F5")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=9)
    ax.set_title("Final Training Metrics")
    ax.set_xlabel("value")
    fig.tight_layout()
    p = out_dir / "final_metrics_bar.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  Saved: {p}")
    return [p]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot training curves from metrics.jsonl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("files", nargs="+",
                        help="One or more metrics.jsonl (or final_metrics.json)")
    parser.add_argument("--out", "-o", default="figures",
                        help="Output directory (default: figures/)")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Labels for each file (for multi-run comparison)")
    parser.add_argument("--window", "-w", type=int, default=5,
                        help="Smoothing window (default: 5)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records: List[List[dict]] = []
    labels: List[str] = args.labels or []

    for i, fpath in enumerate(args.files):
        p = Path(fpath)
        if not p.exists():
            print(f"WARNING: {p} not found, skipping.")
            continue

        if p.name.endswith(".json") and not p.name.endswith(".jsonl"):
            print(f"[{p.name}] Plotting final metrics bar chart...")
            plot_final_bar(p, out_dir)
            continue

        records = load_jsonl(p)
        if not records:
            print(f"WARNING: {p} is empty, skipping.")
            continue

        lbl = labels[i] if i < len(labels) else p.parent.name
        all_records.append(records)
        if len(labels) <= i:
            labels.append(lbl)

        trainer = detect_trainer_type(records)
        print(f"[{p}] Detected trainer: {trainer} ({len(records)} records)")

        if trainer == "adversarial":
            plot_adversarial(records, out_dir, label=lbl, window=args.window)
        elif trainer == "grpo":
            plot_grpo(records, out_dir, label=lbl, window=args.window)
        else:
            plot_reinforce(records, out_dir, label=lbl, window=args.window)

    if len(all_records) > 1:
        print("Plotting multi-run comparison...")
        plot_comparison(all_records, labels, out_dir, window=args.window)

    print(f"\nAll figures saved to: {out_dir}/")


if __name__ == "__main__":
    main()
