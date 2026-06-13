"""Train K-Means and SVM, evaluate them, and cache the fitted bundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from employee_app.config import (
    CATEGORICAL_FEATURES,
    DATASET_PATH,
    MODEL_FEATURES,
    MODEL_PATH,
    N_CLUSTERS,
    NUMERIC_FEATURES,
    PIPELINE_VERSION,
    RANDOM_STATE,
)
from employee_app.data_loader import dataset_hash, load_dataset


@dataclass(frozen=True)
class ModelBundle:
    """All fitted objects and metadata needed during inference."""

    preprocessor: ColumnTransformer
    kmeans: KMeans
    svm: CalibratedClassifierCV
    cluster_labels: dict[int, str]
    cluster_profiles: dict[int, dict[str, float | int | str]]
    metrics: dict[str, float]
    dataset_rows: int
    dataset_hash: str
    pipeline_version: str


def build_preprocessor() -> ColumnTransformer:
    """Create the preprocessing rules used by both training and prediction."""

    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "department",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        sparse_threshold=0,
    )


def label_clusters(dataframe: pd.DataFrame, clusters: np.ndarray) -> dict[int, str]:
    """Assign stable, descriptive names from each cluster's mean attributes."""

    summary = (
        dataframe.assign(cluster=clusters)
        .groupby("cluster")[["Age", "TotalWorkingYears", "Education"]]
        .mean()
    )
    if len(summary) != N_CLUSTERS:
        raise ValueError(f"Diharapkan {N_CLUSTERS} cluster, ditemukan {len(summary)}.")

    seasoned_cluster = int(
        max(
            summary.index,
            key=lambda cluster_id: (
                summary.loc[cluster_id, "TotalWorkingYears"],
                summary.loc[cluster_id, "Age"],
            ),
        )
    )
    remaining = [
        int(cluster_id)
        for cluster_id in summary.index
        if cluster_id != seasoned_cluster
    ]
    academic_cluster = int(
        max(
            remaining,
            key=lambda cluster_id: (
                summary.loc[cluster_id, "Education"],
                -summary.loc[cluster_id, "TotalWorkingYears"],
            ),
        )
    )
    emerging_cluster = next(
        cluster_id for cluster_id in remaining if cluster_id != academic_cluster
    )

    return {
        emerging_cluster: "Emerging Talent",
        academic_cluster: "Academic Achiever",
        seasoned_cluster: "Seasoned Veteran",
    }


def summarize_clusters(
    dataframe: pd.DataFrame,
    clusters: np.ndarray,
    cluster_labels: dict[int, str],
) -> dict[int, dict[str, float | int | str]]:
    """Create readable cluster statistics for the educational dashboard."""

    grouped = dataframe.assign(cluster=clusters).groupby("cluster")
    profiles: dict[int, dict[str, float | int | str]] = {}
    for cluster_id, cluster_data in grouped:
        numeric_means = cluster_data[
            ["Age", "TotalWorkingYears", "Education"]
        ].mean()
        profiles[int(cluster_id)] = {
            "category": cluster_labels[int(cluster_id)],
            "employee_count": int(len(cluster_data)),
            "average_age": round(float(numeric_means["Age"]), 2),
            "average_experience": round(
                float(numeric_means["TotalWorkingYears"]),
                2,
            ),
            "average_education": round(
                float(numeric_means["Education"]),
                2,
            ),
        }
    return profiles


def _new_svc() -> SVC:
    return SVC(
        kernel="rbf",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def train_model(
    dataset_path: Path = DATASET_PATH,
    artifact_path: Path = MODEL_PATH,
) -> ModelBundle:
    dataframe = load_dataset(dataset_path)
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(dataframe[MODEL_FEATURES])

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        n_init=20,
        random_state=RANDOM_STATE,
    )
    cluster_ids = kmeans.fit_predict(transformed)
    cluster_labels = label_clusters(dataframe, cluster_ids)
    cluster_profiles = summarize_clusters(
        dataframe,
        cluster_ids,
        cluster_labels,
    )

    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    cross_validated_predictions = cross_val_predict(
        _new_svc(),
        transformed,
        cluster_ids,
        cv=cross_validation,
    )
    metrics = {
        "svm_accuracy": float(
            accuracy_score(cluster_ids, cross_validated_predictions)
        ),
        "svm_balanced_accuracy": float(
            balanced_accuracy_score(cluster_ids, cross_validated_predictions)
        ),
        "kmeans_silhouette": float(silhouette_score(transformed, cluster_ids)),
    }

    calibrated_svm = CalibratedClassifierCV(
        estimator=_new_svc(),
        method="sigmoid",
        cv=cross_validation,
    )
    calibrated_svm.fit(transformed, cluster_ids)

    bundle = ModelBundle(
        preprocessor=preprocessor,
        kmeans=kmeans,
        svm=calibrated_svm,
        cluster_labels=cluster_labels,
        cluster_profiles=cluster_profiles,
        metrics=metrics,
        dataset_rows=len(dataframe),
        dataset_hash=dataset_hash(dataset_path),
        pipeline_version=PIPELINE_VERSION,
    )

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_path)
    return bundle


def _is_current(bundle: Any, current_hash: str) -> bool:
    return (
        isinstance(bundle, ModelBundle)
        and bundle.pipeline_version == PIPELINE_VERSION
        and bundle.dataset_hash == current_hash
    )


def load_or_train_model(
    dataset_path: Path = DATASET_PATH,
    artifact_path: Path = MODEL_PATH,
) -> tuple[ModelBundle, bool]:
    current_hash = dataset_hash(dataset_path)
    if artifact_path.exists():
        try:
            cached_bundle = joblib.load(artifact_path)
            if _is_current(cached_bundle, current_hash):
                return cached_bundle, True
        except Exception:
            pass

    return train_model(dataset_path, artifact_path), False
