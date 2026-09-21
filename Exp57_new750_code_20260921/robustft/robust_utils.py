"""Pure helpers for robust-training decisions and checkpoint safety."""

from __future__ import annotations

from pathlib import Path


def validate_disjoint_split(train_idx, val_idx) -> None:
    train = {int(i) for i in train_idx}
    val = {int(i) for i in val_idx}
    if not train:
        raise ValueError("training split must not be empty")
    if train & val:
        raise ValueError("training and validation splits overlap")


def reliability_weight(
    agreement: float,
    observed_label_confidence: float,
    prediction_margin: float,
    *,
    agreement_power: float = 1.0,
    confidence_power: float = 0.5,
    margin_power: float = 0.5,
    minimum: float = 0.2,
) -> float:
    score = (
        max(0.0, min(1.0, agreement)) ** agreement_power
        * max(0.0, min(1.0, observed_label_confidence)) ** confidence_power
        * max(0.0, min(1.0, prediction_margin)) ** margin_power
    )
    return max(0.0, min(1.0, minimum + (1.0 - minimum) * score))


def consensus_pseudo_mask(
    teacher_prediction: int,
    knn_prediction: int,
    teacher_confidence: float,
    prediction_margin: float,
    confidence_threshold: float,
    margin_threshold: float,
) -> bool:
    return (
        teacher_prediction == knn_prediction
        and teacher_confidence >= confidence_threshold
        and prediction_margin >= margin_threshold
    )


def choose_checkpoint(checkpoint_dir: Path, policy: str) -> Path:
    if policy not in {"best", "last", "full"}:
        raise ValueError(f"unsupported checkpoint policy: {policy}")
    path = checkpoint_dir / f"{policy}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"requested checkpoint does not exist: {path}")
    return path
