from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

from .metrics import evaluate_classifier


@dataclass
class TrainConfig:
    """
    Training configuration for RandomForest.
    """
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 300
    max_depth: int | None = None
    min_samples_leaf: int = 1
    class_weight: str | dict | None = "balanced"


DEFAULT_META_COLS = ["window_start", "window_end"]
DEFAULT_GROUP_COL = "patient_id"
DEFAULT_TARGET_COL = "risk_level"


def _select_feature_columns(df: pd.DataFrame, target_col: str, include_tags: bool) -> List[str]:
    """
    Decide which columns are used as features.

    - Always exclude the target column.
    - Always exclude obvious metadata columns if present.
    - Exclude tag_* features by default (to avoid rule leakage).
    """
    exclude = {target_col, *DEFAULT_META_COLS}

    # We typically should NOT use patient_id as a feature.
    # It is used only for group-based splitting.
    if DEFAULT_GROUP_COL in df.columns:
        exclude.add(DEFAULT_GROUP_COL)

    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if (not include_tags) and c.startswith("tag_"):
            continue
        cols.append(c)

    return cols


def _prepare_xy_groups(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    include_tags: bool,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Prepare X, y and groups for group-based splitting.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found.")

    # Keep only numeric columns among selected features (safe baseline).
    feature_cols = _select_feature_columns(df, target_col=target_col, include_tags=include_tags)
    X = df[feature_cols].select_dtypes(include=["number"]).copy()

    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns selected.")

    # Make sure the target is numeric/int labels.
    y = df[target_col].to_numpy()
    groups = df[group_col].to_numpy()

    # Replace inf with NaN, then fill NaNs (if any) with 0.0.
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return X, y, groups


def _group_train_test_split(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Split by groups (patients) so the test set contains unseen patients.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    return X_train, X_test, y_train, y_test


def train_random_forest(
    input_csv: str,
    output_dir: str = "data",
    target_col: str = DEFAULT_TARGET_COL,
    group_col: str = DEFAULT_GROUP_COL,
    include_tags: bool = False,
    cfg: Optional[TrainConfig] = None,
) -> None:
    """
    Train RandomForest on Information Arrays with:
      - patient-level (group) split
      - optional inclusion/exclusion of semantic tags

    Saves:
      - rf_model.joblib
      - metrics.json
      - confusion_matrix.csv
    """
    if cfg is None:
        cfg = TrainConfig()

    input_path = Path(input_csv)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    X, y, groups = _prepare_xy_groups(
        df=df,
        target_col=target_col,
        group_col=group_col,
        include_tags=include_tags,
    )

    # Determine label set (ignoring invalid labels like -1).
    labels = sorted({int(v) for v in np.unique(y) if int(v) >= 0})
    if not labels:
        raise ValueError("No valid labels found in target column.")

    # Group-based split: test contains unseen patients.
    X_train, X_test, y_train, y_test = _group_train_test_split(
        X, y, groups,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
    )

    model = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        random_state=cfg.random_state,
        n_jobs=-1,
        class_weight=cfg.class_weight,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    eval_res = evaluate_classifier(y_true=y_test, y_pred=y_pred, labels=labels)

    # Save artifacts
    model_path = out_dir / "rf_model.joblib"
    joblib.dump(model, model_path)

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "RandomForestClassifier",
                "n_features": int(X.shape[1]),
                "feature_names": list(X.columns),
                "config": cfg.__dict__,
                "split": {
                    "strategy": "GroupShuffleSplit",
                    "group_col": group_col,
                    "test_size": cfg.test_size,
                    "random_state": cfg.random_state,
                },
                "include_tags": include_tags,
                "evaluation": eval_res.metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Confusion matrix as CSV
    cm_path = out_dir / "confusion_matrix.csv"
    cm_df = pd.DataFrame(
        eval_res.confusion,
        index=[f"true_{l}" for l in labels],
        columns=[f"pred_{l}" for l in labels],
    )
    cm_df.to_csv(cm_path, index=True)

    # Print a compact summary
    print("=== Training finished ===")
    print(f"Input: {input_path.resolve()}")
    print(f"Saved model: {model_path.resolve()}")
    print(f"Saved metrics: {metrics_path.resolve()}")
    print(f"Saved confusion matrix: {cm_path.resolve()}")
    print(f"Include semantic tags: {include_tags}")
    print(f"Accuracy: {eval_res.metrics['accuracy']:.4f}")
    print(f"Macro F1: {eval_res.metrics['macro_avg']['f1']:.4f}")
