"""K-Means clustering, cluster interpretation, evaluation, and prediction.

Presentation flow:
1. Build a deterministic K-Means model with three clusters.
2. Fit the model to preprocessed employee vectors.
3. Name clusters from their average age, experience, and education.
4. Evaluate cluster separation with the silhouette score.
5. Predict a new vector from its nearest centroid.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from employee_app.core.config import N_CLUSTERS, RANDOM_STATE


def create_kmeans_model() -> KMeans:
    """Create the K-Means configuration used for training."""

    return KMeans(
        n_clusters=N_CLUSTERS,
        n_init=20,
        random_state=RANDOM_STATE,
    )


def train_kmeans(
    transformed_data: np.ndarray,
) -> tuple[KMeans, np.ndarray, float]:
    """Fit K-Means and return the model, cluster IDs, and silhouette score."""

    model = create_kmeans_model()
    cluster_ids = model.fit_predict(transformed_data)
    score = float(silhouette_score(transformed_data, cluster_ids))
    return model, cluster_ids, score


def label_clusters(
    dataframe: pd.DataFrame,
    cluster_ids: np.ndarray,
) -> dict[int, str]:
    """Assign stable profile names from the average values in each cluster."""

    summary = (
        dataframe.assign(cluster=cluster_ids)
        .groupby("cluster")[["Age", "TotalWorkingYears", "Education"]]
        .mean()
    )
    if len(summary) != N_CLUSTERS:
        raise ValueError(f"Diharapkan {N_CLUSTERS} cluster, ditemukan {len(summary)}.")

    # The most experienced cluster represents the senior employee profile.
    seasoned_cluster = int(
        max(
            summary.index,
            key=lambda cluster_id: (
                summary.loc[cluster_id, "TotalWorkingYears"],
                summary.loc[cluster_id, "Age"],
            ),
        )
    )
    remaining_clusters = [
        int(cluster_id)
        for cluster_id in summary.index
        if cluster_id != seasoned_cluster
    ]

    # Of the remaining clusters, the highest education average is academic.
    academic_cluster = int(
        max(
            remaining_clusters,
            key=lambda cluster_id: (
                summary.loc[cluster_id, "Education"],
                -summary.loc[cluster_id, "TotalWorkingYears"],
            ),
        )
    )
    emerging_cluster = next(
        cluster_id
        for cluster_id in remaining_clusters
        if cluster_id != academic_cluster
    )

    return {
        emerging_cluster: "Emerging Talent",
        academic_cluster: "Academic Achiever",
        seasoned_cluster: "Seasoned Veteran",
    }


def summarize_clusters(
    dataframe: pd.DataFrame,
    cluster_ids: np.ndarray,
    cluster_labels: dict[int, str],
) -> dict[int, dict[str, float | int | str]]:
    """Summarize each cluster with values that can be presented in the UI."""

    profiles: dict[int, dict[str, float | int | str]] = {}
    for cluster_id, cluster_data in dataframe.assign(
        cluster=cluster_ids
    ).groupby("cluster"):
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


def predict_cluster(model: KMeans, transformed: np.ndarray) -> int:
    """Return the ID of the nearest K-Means centroid."""

    return int(model.predict(transformed)[0])


def build_kmeans_trace(
    model: KMeans,
    transformed: np.ndarray,
    selected_cluster: int,
    cluster_labels: dict[int, str],
) -> dict[str, Any]:
    """Show the Euclidean distance from an input vector to every centroid."""

    distances = model.transform(transformed)[0]
    ranked_distances = sorted(
        (
            {
                "cluster_id": int(cluster_id),
                "category": cluster_labels[int(cluster_id)],
                "distance": round(float(distance), 4),
            }
            for cluster_id, distance in enumerate(distances)
        ),
        key=lambda item: item["distance"],
    )
    return {
        "method": "Euclidean distance to each centroid",
        "distances": ranked_distances,
        "nearest_cluster_id": selected_cluster,
        "nearest_category": cluster_labels[selected_cluster],
    }


def build_cluster_projection(
    transformed_data: np.ndarray,
    model: KMeans,
    cluster_ids: np.ndarray,
    cluster_labels: dict[int, str],
) -> tuple[PCA, dict[str, Any]]:
    """Project training points and centroids to two dimensions for reporting.

    PCA is only a visualization layer. K-Means training and prediction still
    use the complete transformed feature space.
    """

    pca = PCA(n_components=2)
    projected_points = pca.fit_transform(transformed_data)
    projected_centroids = pca.transform(model.cluster_centers_)

    points = [
        {
            "x": round(float(coordinates[0]), 4),
            "y": round(float(coordinates[1]), 4),
            "cluster_id": int(cluster_id),
        }
        for coordinates, cluster_id in zip(
            projected_points,
            cluster_ids,
            strict=True,
        )
    ]
    centroids = [
        {
            "x": round(float(coordinates[0]), 4),
            "y": round(float(coordinates[1]), 4),
            "cluster_id": int(cluster_id),
            "category": cluster_labels[int(cluster_id)],
        }
        for cluster_id, coordinates in enumerate(projected_centroids)
    ]

    return pca, {
        "method": "PCA 2D projection of preprocessed features",
        "axis_labels": ["Principal Component 1", "Principal Component 2"],
        "explained_variance_ratio": [
            round(float(value), 4) for value in pca.explained_variance_ratio_
        ],
        "points": points,
        "centroids": centroids,
        "note": (
            "PCA digunakan untuk visualisasi laporan. Keputusan K-Means tetap "
            "dihitung pada seluruh fitur hasil preprocessing."
        ),
    }


def project_new_point(
    pca: PCA,
    transformed: np.ndarray,
    selected_cluster: int,
    cluster_labels: dict[int, str],
) -> dict[str, float | int | str]:
    """Place a new employee vector on the same PCA plane as training data."""

    coordinates = pca.transform(transformed)[0]
    return {
        "x": round(float(coordinates[0]), 4),
        "y": round(float(coordinates[1]), 4),
        "cluster_id": selected_cluster,
        "category": cluster_labels[selected_cluster],
    }
