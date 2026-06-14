"""Coordinate preprocessing, model training, evaluation, and artifact caching."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from employee_app.core.config import (
    DATASET_PATH,
    MODEL_FEATURES,
    MODEL_PATH,
    PIPELINE_VERSION,
)
from employee_app.core.data_loader import dataset_hash, load_dataset
from employee_app.core.model_bundle import ModelBundle
from employee_app.core.preprocessing import build_preprocessor
from employee_app.models.kmeans import (
    label_clusters,
    summarize_clusters,
    train_kmeans,
)
from employee_app.models.svm import evaluate_svm, train_svm


def train_model(
    dataset_path: Path = DATASET_PATH,
    artifact_path: Path = MODEL_PATH,
) -> ModelBundle:
    """Run the full training pipeline and save one reusable artifact."""

    dataframe = load_dataset(dataset_path)

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(dataframe[MODEL_FEATURES])

    kmeans, cluster_ids, silhouette = train_kmeans(transformed)
    cluster_labels = label_clusters(dataframe, cluster_ids)
    cluster_profiles = summarize_clusters(
        dataframe,
        cluster_ids,
        cluster_labels,
    )

    metrics = evaluate_svm(transformed, cluster_ids)
    metrics["kmeans_silhouette"] = silhouette
    svm = train_svm(transformed, cluster_ids)

    bundle = ModelBundle(
        preprocessor=preprocessor,
        kmeans=kmeans,
        svm=svm,
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
    """Reuse a current artifact or retrain when its schema/data has changed."""

    current_hash = dataset_hash(dataset_path)
    if artifact_path.exists():
        try:
            cached_bundle = joblib.load(artifact_path)
            if _is_current(cached_bundle, current_hash):
                return cached_bundle, True
        except Exception:
            pass

    return train_model(dataset_path, artifact_path), False
