"""Remembers the most recent predictions in a small CSV file so the Overview page can list them."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src import config as C

HISTORY_PATH = C.PROJECT_DIR / "webapp_data" / "prediction_history.csv"
COLUMNS = ["time", "model", "prediction", "confidence", "review"]
MAX_ROWS = 300
MAX_TEXT = 240


def append(model: str, prediction: str, confidence: Optional[float], review: str, path: Path = None) -> bool:
    """Add one row. Returns False (instead of crashing) if the file cannot be written."""
    path = Path(path or HISTORY_PATH)
    row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "prediction": prediction,
        "confidence": "" if confidence is None else f"{confidence:.4f}",
        "review": " ".join(str(review).split())[:MAX_TEXT],
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        _trim(path)
        return True
    except OSError:
        return False


def _trim(path: Path) -> None:
    df = pd.read_csv(path, keep_default_na=False)
    if len(df) > MAX_ROWS:
        df.tail(MAX_ROWS).to_csv(path, index=False, encoding="utf-8")


def recent(n: int = 8, path: Path = None) -> pd.DataFrame:
    """The last n predictions, newest first (empty table if there are none)."""
    path = Path(path or HISTORY_PATH)
    if not path.is_file():
        return pd.DataFrame(columns=COLUMNS)
    try:
        df = pd.read_csv(path, keep_default_na=False)
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=COLUMNS)
    return df.tail(n).iloc[::-1].reset_index(drop=True)
