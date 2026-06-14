import shutil

import numpy as np

from employee_app.core.config import DATASET_PATH
from employee_app.core.training import load_or_train_model, train_model


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
    assert np.allclose(first.pca.components_, second.pca.components_)
    assert first.cluster_plot == second.cluster_plot
    assert first.svm_plot == second.svm_plot


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

    monkeypatch.setattr("employee_app.core.training.train_model", fake_train)
    _, loaded_from_cache = load_or_train_model(dataset_copy, artifact_path)

    assert loaded_from_cache is False
    assert calls == [(dataset_copy, artifact_path)]
