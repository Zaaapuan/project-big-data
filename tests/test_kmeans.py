import numpy as np

from employee_app.core.config import DATASET_PATH, MODEL_FEATURES
from employee_app.core.data_loader import load_dataset
from employee_app.core.preprocessing import build_preprocessor
from employee_app.models.kmeans import (
    build_kmeans_trace,
    label_clusters,
    predict_cluster,
    summarize_clusters,
    train_kmeans,
)


def test_kmeans_produces_three_named_clusters(model_bundle):
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


def test_kmeans_helpers_cover_training_prediction_and_trace():
    dataframe = load_dataset(DATASET_PATH)
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(dataframe[MODEL_FEATURES])

    model, cluster_ids, silhouette = train_kmeans(transformed)
    labels = label_clusters(dataframe, cluster_ids)
    profiles = summarize_clusters(dataframe, cluster_ids, labels)
    sample = transformed[:1]
    selected_cluster = predict_cluster(model, sample)
    trace = build_kmeans_trace(model, sample, selected_cluster, labels)

    assert len(np.unique(cluster_ids)) == 3
    assert silhouette > 0
    assert set(labels.values()) == {
        "Emerging Talent",
        "Academic Achiever",
        "Seasoned Veteran",
    }
    assert len(profiles) == 3
    assert len(trace["distances"]) == 3
    assert trace["nearest_cluster_id"] == selected_cluster
