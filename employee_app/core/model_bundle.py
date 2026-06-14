"""Serializable objects and metadata required during model inference."""

from dataclasses import dataclass

from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.svm import SVC


@dataclass(frozen=True)
class ModelBundle:
    """Keep the fitted pipeline together in one Joblib artifact."""

    preprocessor: ColumnTransformer
    kmeans: KMeans
    svm: CalibratedClassifierCV
    svm_visualizer: SVC
    pca: PCA
    cluster_plot: dict[str, object]
    svm_plot: dict[str, object]
    cluster_labels: dict[int, str]
    cluster_profiles: dict[int, dict[str, float | int | str]]
    metrics: dict[str, float]
    dataset_rows: int
    dataset_hash: str
    pipeline_version: str
