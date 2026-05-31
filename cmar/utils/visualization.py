from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from cmar.utils.io import ensure_dir


def _save(fig: plt.Figure, output_dir: str | Path, name: str) -> None:
    output = ensure_dir(output_dir)
    fig.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")


def plot_training_log(log_csv: str | Path, output_dir: str | Path) -> None:
    df = pd.read_csv(log_csv)
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
    axes[0].plot(df["epoch"], df["train_loss"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(frameon=False)
    axes[1].plot(df["epoch"], df["val_auc"], color="#16697A")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation AUC")
    _save(fig, output_dir, "training_curves")
    plt.close(fig)


def plot_asymmetric_attack_bars(df: pd.DataFrame, output_dir: str | Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    sns.barplot(data=df, x="condition", y="auc", hue="model", ax=ax)
    ax.set_xlabel("Attack condition")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, ncol=2)
    _save(fig, output_dir, "fig2_asymmetric_attack")
    plt.close(fig)


def plot_rar_curves(df: pd.DataFrame, output_dir: str | Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    sns.lineplot(data=df, x="condition", y="rar", hue="model", marker="o", ax=ax)
    ax.set_xlabel("Degradation condition")
    ax.set_ylabel("RAR")
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False, ncol=2)
    _save(fig, output_dir, "fig3_rar_curves")
    plt.close(fig)


def plot_category_auc(df: pd.DataFrame, output_dir: str | Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    sns.barplot(data=df, y="category", x="auc", hue="model", orient="h", ax=ax)
    ax.set_xlabel("AUC")
    ax.set_ylabel("AV category")
    ax.set_xlim(0.0, 1.0)
    ax.legend(frameon=False)
    _save(fig, output_dir, "fig4_category_auc")
    plt.close(fig)


def plot_attention_heatmap(attention, output_dir: str | Path, name: str = "fig5_attention") -> None:
    sns.set_theme(style="white", context="paper")
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    sns.heatmap(attention, cmap="mako", ax=ax, cbar_kws={"label": "attention"})
    ax.set_xlabel("Attended segment")
    ax.set_ylabel("Query segment/head")
    _save(fig, output_dir, name)
    plt.close(fig)


def plot_ablation_bars(df: pd.DataFrame, output_dir: str | Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    sns.barplot(data=df, x="condition", y="auc", hue="model", ax=ax)
    ax.set_xlabel("Condition")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, ncol=2)
    _save(fig, output_dir, "fig6_ablations")
    plt.close(fig)
