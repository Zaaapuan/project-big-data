"""SVM training, cross-validation, probability calibration, and prediction.

Presentation flow:
1. Use K-Means cluster IDs as supervised labels.
2. Evaluate an RBF SVM with stratified five-fold cross-validation.
3. Calibrate the final classifier so it can return profile confidence.
4. Predict a cluster label and explain every category probability.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC

from employee_app.core.config import RANDOM_STATE


def create_svm_model() -> SVC:
    """Create the base RBF classifier used in evaluation and calibration."""

    return SVC(
        kernel="rbf",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def create_cross_validation() -> StratifiedKFold:
    """Use the same deterministic folds for evaluation and calibration."""

    return StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def evaluate_svm(
    transformed_data: np.ndarray,
    cluster_ids: np.ndarray,
) -> dict[str, float]:
    """Measure how consistently SVM reproduces the K-Means cluster labels."""

    predictions = cross_val_predict(
        create_svm_model(),
        transformed_data,
        cluster_ids,
        cv=create_cross_validation(),
    )
    return {
        "svm_accuracy": float(accuracy_score(cluster_ids, predictions)),
        "svm_balanced_accuracy": float(
            balanced_accuracy_score(cluster_ids, predictions)
        ),
    }


def train_svm(
    transformed_data: np.ndarray,
    cluster_ids: np.ndarray,
) -> CalibratedClassifierCV:
    """Train a calibrated RBF SVM so prediction confidence is available."""

    model = CalibratedClassifierCV(
        estimator=create_svm_model(),
        method="sigmoid",
        cv=create_cross_validation(),
    )
    model.fit(transformed_data, cluster_ids)
    return model


def predict_with_probabilities(
    model: CalibratedClassifierCV,
    transformed: np.ndarray,
) -> tuple[int, dict[int, float]]:
    """Return the selected cluster and calibrated probability per cluster."""

    selected_cluster = int(model.predict(transformed)[0])
    probabilities = model.predict_proba(transformed)[0]
    class_probabilities = {
        int(class_id): float(probability)
        for class_id, probability in zip(
            model.classes_,
            probabilities,
            strict=True,
        )
    }
    return selected_cluster, class_probabilities


def build_svm_trace(
    class_probabilities: dict[int, float],
    selected_cluster: int,
    cluster_labels: dict[int, str],
) -> dict[str, Any]:
    """Present calibrated SVM probabilities from highest to lowest."""

    probabilities = sorted(
        (
            {
                "cluster_id": cluster_id,
                "category": cluster_labels[cluster_id],
                "probability": round(probability, 4),
            }
            for cluster_id, probability in class_probabilities.items()
        ),
        key=lambda item: item["probability"],
        reverse=True,
    )
    return {
        "method": "RBF SVM with sigmoid probability calibration",
        "probabilities": probabilities,
        "selected_cluster_id": selected_cluster,
        "selected_category": cluster_labels[selected_cluster],
    }
