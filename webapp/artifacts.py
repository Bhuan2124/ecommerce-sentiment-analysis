"""Reads the files written by the earlier phases. Nothing in here trains or changes anything.

Every loader returns None when its file does not exist, so the pages can show a friendly
"not ready yet" message instead of crashing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src import config as C

# name -> file. (Tests replace this dictionary to point at temporary folders.)
ARTIFACTS: Dict[str, Path] = {
    "dataset_summary": C.RESULTS_DIR / "dataset_summary.json",
    "classical_val": C.RESULTS_DIR / "classical_val_results.csv",
    "classical_summary": C.RESULTS_DIR / "classical_summary.json",
    "classical_tuning": C.RESULTS_DIR / "classical_tuning_all_settings.csv",
    "distilbert_summary": C.RESULTS_DIR / "distilbert_summary.json",
    "distilbert_search": C.RESULTS_DIR / "distilbert_search.csv",
    "distilbert_history": C.RESULTS_DIR / "distilbert_history.csv",
    "training_log": C.RESULTS_DIR / "distilbert_training_log.txt",
    "test_results": C.RESULTS_DIR / "test_results.json",
    "train": C.PROCESSED_DIR / "train.csv",
    "val": C.PROCESSED_DIR / "val.csv",
    "test": C.PROCESSED_DIR / "test.csv",
}

SPLIT_REQUIRED_COLUMNS = [C.TEXT_COL, C.RATING_COL, C.LABEL_COL, C.LABEL_ID_COL]


def path_of(name: str) -> Path:
    return Path(ARTIFACTS[name])


def exists(name: str) -> bool:
    return path_of(name).is_file()


def mtime_key(name: str) -> int:
    """Changes whenever the file changes (used so cached data refreshes after a new run)."""
    p = path_of(name)
    try:
        return p.stat().st_mtime_ns
    except OSError:
        return 0


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------
def _read_json(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _read_csv(path: Path, **kwargs) -> Optional[pd.DataFrame]:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None


def dataset_summary() -> Optional[dict]:
    return _read_json(path_of("dataset_summary"))


def classical_val() -> Optional[pd.DataFrame]:
    return _read_csv(path_of("classical_val"))


def classical_summary() -> Optional[dict]:
    return _read_json(path_of("classical_summary"))


def classical_tuning() -> Optional[pd.DataFrame]:
    return _read_csv(path_of("classical_tuning"))


def distilbert_summary() -> Optional[dict]:
    return _read_json(path_of("distilbert_summary"))


def distilbert_search() -> Optional[pd.DataFrame]:
    return _read_csv(path_of("distilbert_search"))


def distilbert_history() -> Optional[pd.DataFrame]:
    return _read_csv(path_of("distilbert_history"))


def training_log() -> Optional[str]:
    p = path_of("training_log")
    if not p.is_file():
        return None
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def validate_test_results(data: object) -> List[str]:
    """Return a list of problems (empty list = the file looks right)."""
    problems: List[str] = []
    if not isinstance(data, dict):
        return ["test_results.json is not a JSON object"]
    for key in ("labels", "n_test", "models"):
        if key not in data:
            problems.append(f"missing key '{key}'")
    models = data.get("models")
    if isinstance(models, dict):
        for name, entry in models.items():
            if not isinstance(entry, dict):
                problems.append(f"model '{name}' is not an object")
                continue
            if entry.get("status") == "evaluated":
                for key in ("metrics", "per_class", "confusion_matrix"):
                    if key not in entry:
                        problems.append(f"model '{name}' is missing '{key}'")
    return problems


def test_results() -> Optional[dict]:
    """Test-set results written by scripts/evaluate_models.py (Phase 4). None if absent or malformed."""
    data = _read_json(path_of("test_results"))
    if data is None or validate_test_results(data):
        return None
    return data


def load_split(name: str) -> Optional[pd.DataFrame]:
    """One of the processed splits ('train', 'val', 'test'). None if missing or the columns are wrong."""
    df = _read_csv(path_of(name), keep_default_na=False)
    if df is None or validate_split(df):
        return None
    return df


def validate_split(df: pd.DataFrame) -> List[str]:
    """Check a split has the expected columns and consistent labels."""
    problems = [f"missing column '{c}'" for c in SPLIT_REQUIRED_COLUMNS if c not in df.columns]
    if problems:
        return problems
    expected = df[C.RATING_COL].map(C.RATING_TO_LABEL)
    if (expected != df[C.LABEL_COL]).any():
        problems.append("some labels do not match the star-rating mapping")
    if (df[C.LABEL_COL].map(C.LABEL2ID) != df[C.LABEL_ID_COL]).any():
        problems.append("some label ids do not match the label names")
    return problems


def find_raw_csv() -> Optional[Path]:
    files = sorted(Path(C.RAW_DIR).glob("*.csv")) if Path(C.RAW_DIR).is_dir() else []
    return files[0] if files else None


def load_raw() -> Optional[pd.DataFrame]:
    p = find_raw_csv()
    return _read_csv(p) if p else None


def column_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Column name, type, how many values are filled in, how many are missing, how many different values."""
    n = len(df)
    rows = []
    for col in df.columns:
        missing = int(df[col].isna().sum())
        rows.append(
            {
                "column": col,
                "type": str(df[col].dtype),
                "filled": n - missing,
                "missing": missing,
                "missing %": round(100 * missing / n, 2) if n else 0.0,
                "different values": int(df[col].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def inventory() -> pd.DataFrame:
    """Which project files exist (shown on the About page)."""
    rows = []
    items = [(name, path_of(name)) for name in ARTIFACTS]
    items += [
        ("model: logreg", C.CLASSICAL_MODELS_DIR / "logreg.joblib"),
        ("model: naive_bayes", C.CLASSICAL_MODELS_DIR / "naive_bayes.joblib"),
        ("model: linear_svm", C.CLASSICAL_MODELS_DIR / "linear_svm.joblib"),
        ("model: distilbert", C.TRANSFORMER_DIR / "config.json"),
    ]
    for name, p in items:
        p = Path(p)
        try:
            rel = p.relative_to(C.PROJECT_DIR)
        except ValueError:
            rel = p
        rows.append(
            {
                "item": name,
                "file": str(rel).replace("\\", "/"),
                "found": "yes" if p.is_file() else "no",
                "size (KB)": round(p.stat().st_size / 1024, 1) if p.is_file() else None,
            }
        )
    return pd.DataFrame(rows)
