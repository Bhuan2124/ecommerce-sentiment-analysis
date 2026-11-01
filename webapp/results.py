"""Turns the saved result files into the small tables the pages display.

Only real, saved numbers are used here. If something has not been measured yet, it is simply absent
(or marked unavailable) and the pages say so.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import pandas as pd

from src import config as C

KEYS = ["logreg", "naive_bayes", "linear_svm", "distilbert"]
DISPLAY: Dict[str, str] = {**C.CLASSICAL_MODEL_NAMES, "distilbert": C.TRANSFORMER_DISPLAY_NAME}
NAME_TO_KEY = {name: key for key, name in DISPLAY.items()}
FAMILY = {"logreg": "classical", "naive_bayes": "classical", "linear_svm": "classical", "distilbert": "transformer"}
BASELINE_NAME = "Majority baseline"

TABLE_COLUMNS = ["key", "model", "family", "accuracy", "macro_precision", "macro_recall", "macro_f1"]


def class_distribution(dataset_summary: Optional[dict]) -> Optional[pd.DataFrame]:
    """Label, count and percent of the cleaned dataset (negative, neutral, positive)."""
    if not dataset_summary or "class_counts_after_cleaning" not in dataset_summary:
        return None
    counts = dataset_summary["class_counts_after_cleaning"]
    df = pd.DataFrame({"label": C.LABEL_NAMES, "count": [int(counts.get(n, 0)) for n in C.LABEL_NAMES]})
    total = df["count"].sum()
    df["percent"] = df["count"] / total * 100 if total else 0.0
    return df


def total_reviews(dataset_summary: Optional[dict]) -> Optional[int]:
    if not dataset_summary:
        return None
    dedup = dataset_summary.get("deduplication", {})
    return dedup.get("rows_out") or sum(dataset_summary.get("class_counts_after_cleaning", {}).values()) or None


def evaluated_models(test: Optional[dict]) -> Dict[str, dict]:
    """Models that have a finished test-set evaluation, in the project's fixed order."""
    if not test:
        return {}
    models = test.get("models", {})
    return {k: models[k] for k in KEYS if k in models and models[k].get("status") == "evaluated"}


def score_table(
    test: Optional[dict], val_df: Optional[pd.DataFrame], distilbert_summary: Optional[dict]
) -> Tuple[Optional[str], pd.DataFrame]:
    """The best available accuracy / macro-F1 table.

    Returns (split, table) where split is 'test' (final results), 'validation' (earlier look,
    used only until the test evaluation has been run) or None (nothing finished yet).
    """
    ev = evaluated_models(test)
    if ev:
        rows = []
        for key, entry in ev.items():
            m = entry["metrics"]
            rows.append(
                {
                    "key": key,
                    "model": DISPLAY[key],
                    "family": FAMILY[key],
                    "accuracy": m["accuracy"],
                    "macro_precision": m.get("macro_precision"),
                    "macro_recall": m.get("macro_recall"),
                    "macro_f1": m["macro_f1"],
                }
            )
        return "test", pd.DataFrame(rows, columns=TABLE_COLUMNS)

    rows = []
    if val_df is not None and "model" in val_df.columns:
        for _, r in val_df.iterrows():
            key = NAME_TO_KEY.get(str(r["model"]))
            if key is None:  # e.g. the majority baseline
                continue
            rows.append({"key": key, "model": DISPLAY[key], "family": FAMILY[key], "accuracy": float(r["accuracy"]),
                         "macro_precision": None, "macro_recall": None, "macro_f1": float(r["macro_f1"])})
    if distilbert_summary and distilbert_summary.get("val_metrics"):
        m = distilbert_summary["val_metrics"]
        rows.append({"key": "distilbert", "model": DISPLAY["distilbert"], "family": "transformer",
                     "accuracy": float(m["accuracy"]), "macro_precision": None, "macro_recall": None,
                     "macro_f1": float(m["macro_f1"])})
    if rows:
        return "validation", pd.DataFrame(rows, columns=TABLE_COLUMNS)
    return None, pd.DataFrame(columns=TABLE_COLUMNS)


def best_model(table: pd.DataFrame) -> Optional[pd.Series]:
    """Row with the highest macro-F1 (ties: the first one)."""
    if table is None or table.empty:
        return None
    return table.sort_values("macro_f1", ascending=False).iloc[0]


def cost_table(test: Optional[dict]) -> pd.DataFrame:
    """Training time, inference time, size, parameters for every evaluated model."""
    rows = []
    for key, entry in evaluated_models(test).items():
        cost = entry.get("cost", {})
        rows.append(
            {
                "key": key,
                "model": DISPLAY[key],
                "train_seconds": cost.get("train_seconds"),
                "train_note": cost.get("train_note", ""),
                "inference_ms_per_review": cost.get("inference_ms_per_review"),
                "inference_device": cost.get("inference_device", ""),
                "model_size_mb": cost.get("model_size_mb"),
                "n_parameters": cost.get("n_parameters"),
            }
        )
    return pd.DataFrame(rows)


def per_class_table(test: Optional[dict]) -> pd.DataFrame:
    """Long table: model, class, precision, recall, f1, support."""
    rows = []
    for key, entry in evaluated_models(test).items():
        for label in C.LABEL_NAMES:
            pc = entry["per_class"].get(label)
            if pc:
                rows.append({"key": key, "model": DISPLAY[key], "class": label, **pc})
    return pd.DataFrame(rows)


def human_time(seconds: Optional[float]) -> str:
    if seconds is None:
        return "n/a"
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 90:
        return f"{seconds:.1f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def human_count(n: Optional[float]) -> str:
    if n is None:
        return "n/a"
    n = float(n)
    if n >= 1e6:
        return f"{n / 1e6:.1f} M"
    if n >= 1e3:
        return f"{n / 1e3:.1f} K"
    return f"{n:.0f}"
