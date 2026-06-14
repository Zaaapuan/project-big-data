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


def build_svm_decision_projection(
    projected_data: np.ndarray,
    cluster_ids: np.ndarray,
    cluster_labels: dict[int, str],
) -> tuple[SVC, dict[str, Any]]:
    """Train an RBF SVC on PCA coordinates for a 2D decision-boundary plot.

    This classifier is a reporting aid only. The production prediction still
    uses the calibrated model trained on the complete transformed feature set.
    """

    projected_data = np.asarray(projected_data)
    cluster_ids = np.asarray(cluster_ids)
    visualizer = create_svm_model()
    visualizer.fit(projected_data, cluster_ids)

    x_padding = float(np.ptp(projected_data[:, 0]) * 0.08) or 1.0
    y_padding = float(np.ptp(projected_data[:, 1]) * 0.10) or 1.0
    x_values = np.linspace(
        float(projected_data[:, 0].min()) - x_padding,
        float(projected_data[:, 0].max()) + x_padding,
        100,
    )
    y_values = np.linspace(
        float(projected_data[:, 1].min()) - y_padding,
        float(projected_data[:, 1].max()) + y_padding,
        60,
    )
    mesh_x, mesh_y = np.meshgrid(x_values, y_values)
    mesh_points = np.column_stack((mesh_x.ravel(), mesh_y.ravel()))
    mesh_predictions = visualizer.predict(mesh_points)

    support_vectors = [
        {
            "x": round(float(point[0]), 4),
            "y": round(float(point[1]), 4),
        }
        for point in visualizer.support_vectors_
    ]
    return visualizer, {
        "method": "SVC with RBF kernel on PCA 2D coordinates",
        "kernel": "rbf",
        "c": float(visualizer.C),
        "gamma": round(float(visualizer._gamma), 6),
        "axis_labels": ["Principal Component 1", "Principal Component 2"],
        "grid": {
            "x_values": [round(float(value), 4) for value in x_values],
            "y_values": [round(float(value), 4) for value in y_values],
            "cluster_ids": [int(value) for value in mesh_predictions],
        },
        "support_vectors": support_vectors,
        "categories": [
            {
                "cluster_id": int(cluster_id),
                "category": cluster_labels[int(cluster_id)],
            }
            for cluster_id in sorted(cluster_labels)
        ],
        "note": (
            "2D boundary untuk melihat pola kelas. Main prediction memakai "
            "calibrated RBF SVM."
        ),
    }


def project_svm_new_point(
    visualizer: SVC,
    projected_point: np.ndarray,
    selected_cluster: int,
    cluster_labels: dict[int, str],
) -> dict[str, float | int | str | bool]:
    """Compare the 2D visual boundary with the complete SVM prediction."""

    visual_cluster = int(visualizer.predict(projected_point)[0])
    return {
        "x": round(float(projected_point[0, 0]), 4),
        "y": round(float(projected_point[0, 1]), 4),
        "cluster_id": selected_cluster,
        "category": cluster_labels[selected_cluster],
        "visual_cluster_id": visual_cluster,
        "visual_category": cluster_labels[visual_cluster],
        "visual_matches_main_model": visual_cluster == selected_cluster,
    }


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
