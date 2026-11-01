"""Loads the saved models and predicts. Used by the Sentiment Analyzer page.

Rules
  * Models are only LOADED from disk here. Nothing is ever trained or re-fitted.
  * A missing model file raises ModelUnavailableError; a file that exists but cannot be read raises
    ModelLoadError; bad user input raises InvalidInputError. The page turns each into a friendly message.
  * Scores are labelled by what they really are:
        probability      Logistic Regression, Naive Bayes (predict_proba)
        decision_score   Linear SVM (distance from the separating boundary; NOT a probability)
        softmax          DistilBERT (softmax of the output layer; NOT necessarily a calibrated probability)
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src import config as C

MAX_INPUT_CHARS = 5000


class InvalidInputError(ValueError):
    """The text typed by the user cannot be analysed."""


class ModelUnavailableError(Exception):
    """The model files are not there (not trained yet, or still training)."""


class ModelLoadError(Exception):
    """The model files exist but loading failed."""


@dataclass(frozen=True)
class ModelInfo:
    key: str
    name: str
    family: str          # "classical" or "transformer"
    score_kind: str      # "probability" | "decision_score" | "softmax"
    blurb: str
    how_to_create: str


MODEL_INFOS: Dict[str, ModelInfo] = {
    "logreg": ModelInfo(
        "logreg", C.CLASSICAL_MODEL_NAMES["logreg"], "classical", "probability",
        "TF-IDF + Logistic Regression", "Run notebooks/02_classical_models.ipynb to train and save it.",
    ),
    "naive_bayes": ModelInfo(
        "naive_bayes", C.CLASSICAL_MODEL_NAMES["naive_bayes"], "classical", "probability",
        "TF-IDF + Multinomial Naive Bayes", "Run notebooks/02_classical_models.ipynb to train and save it.",
    ),
    "linear_svm": ModelInfo(
        "linear_svm", C.CLASSICAL_MODEL_NAMES["linear_svm"], "classical", "decision_score",
        "TF-IDF + Linear SVM", "Run notebooks/02_classical_models.ipynb to train and save it.",
    ),
    "distilbert": ModelInfo(
        "distilbert", C.TRANSFORMER_DISPLAY_NAME, "transformer", "softmax",
        "Fine-tuned DistilBERT",
        "Run notebooks/03_distilbert.ipynb (it saves the model to models/distilbert/). "
        "If training is still running, wait for it to finish.",
    ),
}
MODEL_ORDER = ["logreg", "naive_bayes", "linear_svm", "distilbert"]


@dataclass
class Prediction:
    model_key: str
    label_id: int
    label: str
    scores: Dict[str, float]
    score_kind: str
    confidence: Optional[float]   # only for probability-like scores; None for the SVM
    latency_ms: float
    truncated: bool = False
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_key": self.model_key, "label_id": self.label_id, "label": self.label, "scores": self.scores,
            "score_kind": self.score_kind, "confidence": self.confidence, "latency_ms": self.latency_ms,
            "truncated": self.truncated,
        }


# ---------------------------------------------------------------------------
# Input checking
# ---------------------------------------------------------------------------
def validate_text(text, max_chars: int = MAX_INPUT_CHARS) -> str:
    """Return the cleaned review text, or raise InvalidInputError with a friendly message."""
    if text is None or not isinstance(text, str):
        raise InvalidInputError("Please type or paste a review first.")
    cleaned = " ".join(text.split())
    if not cleaned:
        raise InvalidInputError("The review box is empty. Type or paste a review, then press Analyze.")
    if not any(ch.isalpha() for ch in cleaned):
        raise InvalidInputError("That text has no letters in it. Please enter a review written in words.")
    if len(cleaned) > max_chars:
        raise InvalidInputError(
            f"That review is {len(cleaned):,} characters long. The limit is {max_chars:,}. Please shorten it."
        )
    return cleaned


# ---------------------------------------------------------------------------
# Where the models live, and whether they are there
# ---------------------------------------------------------------------------
def _dirs(classical_dir, distilbert_dir) -> Tuple[Path, Path]:
    return Path(classical_dir or C.CLASSICAL_MODELS_DIR), Path(distilbert_dir or C.TRANSFORMER_DIR)


def availability(key: str, classical_dir=None, distilbert_dir=None) -> Tuple[bool, str]:
    """(is_available, reason_if_not). Only checks files; nothing is imported or loaded."""
    if key not in MODEL_INFOS:
        return False, f"Unknown model '{key}'."
    info = MODEL_INFOS[key]
    cdir, ddir = _dirs(classical_dir, distilbert_dir)
    if info.family == "classical":
        path = cdir / f"{key}.joblib"
        if not path.is_file():
            return False, f"The saved model file {_short(path)} was not found. {info.how_to_create}"
        return True, ""
    has_config = (ddir / "config.json").is_file()
    has_weights = (ddir / "model.safetensors").is_file() or (ddir / "pytorch_model.bin").is_file()
    has_tokenizer = (ddir / "tokenizer.json").is_file() or (ddir / "vocab.txt").is_file()
    if not (has_config and has_weights and has_tokenizer):
        return False, f"The fine-tuned model folder {_short(ddir)} is missing or incomplete. {info.how_to_create}"
    return True, ""


def _short(path: Path) -> str:
    try:
        return str(Path(path).relative_to(C.PROJECT_DIR)).replace("\\", "/")
    except ValueError:
        return str(path)


def available_models(classical_dir=None, distilbert_dir=None) -> Dict[str, bool]:
    return {k: availability(k, classical_dir, distilbert_dir)[0] for k in MODEL_ORDER}


# ---------------------------------------------------------------------------
# Loaded models
# ---------------------------------------------------------------------------
class ClassicalModel:
    """A saved scikit-learn pipeline: TF-IDF vectorizer + classifier."""

    def __init__(self, key: str, pipeline):
        self.key = key
        self.info = MODEL_INFOS[key]
        self.pipeline = pipeline
        classes = [int(c) for c in getattr(pipeline, "classes_", [])]
        if classes != list(range(len(C.LABEL_NAMES))):
            raise ModelLoadError(
                f"The saved {self.info.name} model uses classes {classes}, expected {list(range(len(C.LABEL_NAMES)))}."
            )

    def predict_batch(self, texts: Sequence[str]) -> List[Prediction]:
        texts = list(texts)
        t0 = time.perf_counter()
        last_step = self.pipeline.steps[-1][1]
        if hasattr(last_step, "predict_proba"):
            raw = np.asarray(self.pipeline.predict_proba(texts), dtype=float)
            kind = "probability"
        else:
            raw = np.asarray(self.pipeline.decision_function(texts), dtype=float)
            kind = "decision_score"
        per_review_ms = (time.perf_counter() - t0) * 1000 / max(1, len(texts))
        out = []
        for row in raw:
            idx = int(np.argmax(row))
            out.append(
                Prediction(
                    self.key, idx, C.ID2LABEL[idx], {C.ID2LABEL[i]: float(v) for i, v in enumerate(row)}, kind,
                    float(row[idx]) if kind == "probability" else None, per_review_ms,
                )
            )
        return out


class DistilBertModel:
    """The fine-tuned DistilBERT (Hugging Face format). Runs on the CPU unless SENTIMENT_DEVICE=cuda."""

    def __init__(self, tokenizer, model, torch_module, device: str):
        self.key = "distilbert"
        self.info = MODEL_INFOS["distilbert"]
        self.tokenizer, self.model, self.torch, self.device = tokenizer, model, torch_module, device
        self.max_length = C.MAX_LENGTH

    @classmethod
    def load(cls, directory: Path, device: Optional[str] = None) -> "DistilBertModel":
        import torch  # imported here so the app starts quickly and works without torch installed
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = device or os.environ.get("SENTIMENT_DEVICE", "cpu")
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        tokenizer = AutoTokenizer.from_pretrained(str(directory))
        model = AutoModelForSequenceClassification.from_pretrained(str(directory)).to(device).eval()
        id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
        default_names = {i: f"LABEL_{i}" for i in range(len(id2label))}
        expected = {i: n for i, n in enumerate(C.LABEL_NAMES)}
        if id2label not in (expected, default_names) or len(id2label) != len(C.LABEL_NAMES):
            raise ModelLoadError(f"The saved DistilBERT has labels {id2label}, expected {expected}.")
        return cls(tokenizer, model, torch, device)

    def predict_batch(self, texts: Sequence[str]) -> List[Prediction]:
        texts = list(texts)
        torch = self.torch
        t0 = time.perf_counter()
        enc = self.tokenizer(
            texts, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.device)
        with torch.inference_mode():
            logits = self.model(**enc).logits
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        per_review_ms = (time.perf_counter() - t0) * 1000 / max(1, len(texts))
        full_lengths = [len(ids) for ids in self.tokenizer(texts, truncation=False)["input_ids"]]
        out = []
        for row, n_tokens in zip(probs, full_lengths):
            idx = int(np.argmax(row))
            out.append(
                Prediction(
                    "distilbert", idx, C.ID2LABEL[idx], {C.ID2LABEL[i]: float(v) for i, v in enumerate(row)},
                    "softmax", float(row[idx]), per_review_ms, truncated=n_tokens > self.max_length,
                    extra={"tokens": int(n_tokens), "device": self.device},
                )
            )
        return out


LoadedModel = object  # ClassicalModel or DistilBertModel (both have .predict_batch(texts))


def load_model(key: str, classical_dir=None, distilbert_dir=None, device: Optional[str] = None):
    """Load one saved model from disk. Raises ModelUnavailableError or ModelLoadError."""
    ok, reason = availability(key, classical_dir, distilbert_dir)
    if not ok:
        raise ModelUnavailableError(reason)
    cdir, ddir = _dirs(classical_dir, distilbert_dir)
    try:
        if MODEL_INFOS[key].family == "classical":
            import joblib

            return ClassicalModel(key, joblib.load(cdir / f"{key}.joblib"))
        return DistilBertModel.load(ddir, device)
    except (ModelUnavailableError, ModelLoadError):
        raise
    except Exception as exc:  # corrupt file, version mismatch, missing library ...
        raise ModelLoadError(f"{MODEL_INFOS[key].name} could not be loaded ({type(exc).__name__}: {exc}).") from exc


def predict_text(model, text) -> Prediction:
    """Check the text, then run the model on it."""
    return model.predict_batch([validate_text(text)])[0]
