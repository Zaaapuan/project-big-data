import shutil

import numpy as np

from employee_app.config import DATASET_PATH
from employee_app.model_trainer import (
    load_or_train_model,
    train_model,
)


def test_cluster_labels_are_complete_and_unique(model_bundle):
    assert set(model_bundle.cluster_labels) == {0, 1, 2}
    assert set(model_bundle.cluster_labels.values()) == {
        "Emerging Talent",
        "Academic Achiever",
        "Seasoned Veteran",
    }
    assert len(model_bundle.cluster_profiles) == 3
    assert all(
        profile["employee_count"] > 0
        for profile in model_bundle.cluster_profiles.values()
    )


def test_svm_consistency_exceeds_acceptance_threshold(model_bundle):
    assert model_bundle.metrics["svm_balanced_accuracy"] >= 0.90


def test_training_is_deterministic(tmp_path):
    first = train_model(DATASET_PATH, tmp_path / "first.joblib")
    second = train_model(DATASET_PATH, tmp_path / "second.joblib")

    assert first.cluster_labels == second.cluster_labels
    assert np.allclose(
        first.kmeans.cluster_centers_,
        second.kmeans.cluster_centers_,
    )
    assert first.metrics == second.metrics
    assert first.cluster_profiles == second.cluster_profiles


def test_cache_is_reused_and_invalidated_by_dataset_change(tmp_path, monkeypatch):
    dataset_copy = tmp_path / "employees.csv"
    artifact_path = tmp_path / "model.joblib"
    shutil.copyfile(DATASET_PATH, dataset_copy)

    original_bundle = train_model(dataset_copy, artifact_path)
    cached_bundle, loaded_from_cache = load_or_train_model(
        dataset_copy,
        artifact_path,
    )
    assert loaded_from_cache is True
    assert cached_bundle.dataset_hash == original_bundle.dataset_hash

    with dataset_copy.open("a", encoding="utf-8") as dataset_file:
        dataset_file.write("\n")

    calls = []

    def fake_train(dataset_path, model_path):
        calls.append((dataset_path, model_path))
        return original_bundle

    monkeypatch.setattr("employee_app.model_trainer.train_model", fake_train)
    _, loaded_from_cache = load_or_train_model(dataset_copy, artifact_path)

    assert loaded_from_cache is False
    assert calls == [(dataset_copy, artifact_path)]
