"""Validate employee input and coordinate the explainable prediction flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from employee_app.core.config import (
    CATEGORY_METADATA,
    DEPARTMENTS,
    EDUCATION_LEVELS,
)
from employee_app.core.model_bundle import ModelBundle
from employee_app.core.preprocessing import (
    build_preprocessing_trace,
    employee_frame,
)
from employee_app.models.kmeans import (
    build_kmeans_trace,
    predict_cluster,
    project_new_point,
)
from employee_app.models.svm import (
    build_svm_trace,
    predict_with_probabilities,
    project_svm_new_point,
)


class InputValidationError(ValueError):
    def __init__(self, errors: dict[str, str]):
        super().__init__("Data input tidak valid.")
        self.errors = errors


@dataclass(frozen=True)
class EmployeeInput:
    age: int
    years_experience: int
    education_level: int
    department: str

    def as_model_frame(self):
        return employee_frame(
            self.age,
            self.years_experience,
            self.education_level,
            self.department,
        )


def _parse_integer(
    payload: dict[str, Any],
    field: str,
    label: str,
    minimum: int,
    maximum: int,
    errors: dict[str, str],
) -> int | None:
    value = payload.get(field)
    if value is None or value == "":
        errors[field] = f"{label} wajib diisi."
        return None
    if isinstance(value, bool):
        errors[field] = f"{label} harus berupa bilangan bulat."
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        errors[field] = f"{label} harus berupa bilangan bulat."
        return None
    if isinstance(value, float) and not value.is_integer():
        errors[field] = f"{label} harus berupa bilangan bulat."
        return None
    if not minimum <= parsed <= maximum:
        errors[field] = f"{label} harus berada pada rentang {minimum}-{maximum}."
        return None
    return parsed


def validate_employee_input(payload: Any) -> EmployeeInput:
    """Validate the API payload before any model transformation is performed."""

    if not isinstance(payload, dict):
        raise InputValidationError({"request": "Body harus berupa objek JSON."})

    errors: dict[str, str] = {}
    age = _parse_integer(payload, "age", "Umur", 18, 60, errors)
    experience = _parse_integer(
        payload,
        "years_experience",
        "Tahun pengalaman",
        0,
        40,
        errors,
    )
    education = _parse_integer(
        payload,
        "education_level",
        "Tingkat pendidikan",
        1,
        5,
        errors,
    )

    department = payload.get("department")
    if not department:
        errors["department"] = "Departemen wajib dipilih."
    elif department not in DEPARTMENTS:
        errors["department"] = "Departemen tidak dikenali."

    if age is not None and experience is not None and experience > age - 14:
        errors["years_experience"] = (
            "Tahun pengalaman tidak boleh melebihi umur dikurangi 14 tahun."
        )

    if errors:
        raise InputValidationError(errors)

    return EmployeeInput(
        age=age,
        years_experience=experience,
        education_level=education,
        department=department,
    )


class EmployeeProfilePredictor:
    """Coordinate preprocessing and the two independent model modules."""

    def __init__(self, bundle: ModelBundle, loaded_from_cache: bool = False):
        self.bundle = bundle
        self.loaded_from_cache = loaded_from_cache

    def predict(self, payload: Any) -> dict[str, Any]:
        """Return model results plus the educational trace used by the wizard."""

        employee = validate_employee_input(payload)
        transformed = self.bundle.preprocessor.transform(employee.as_model_frame())

        kmeans_cluster = predict_cluster(self.bundle.kmeans, transformed)
        svm_cluster, class_probabilities = predict_with_probabilities(
            self.bundle.svm,
            transformed,
        )

        svm_category = self.bundle.cluster_labels[svm_cluster]
        kmeans_category = self.bundle.cluster_labels[kmeans_cluster]
        metadata = CATEGORY_METADATA[svm_category]
        projected_point = self.bundle.pca.transform(transformed)

        return {
            "category": svm_category,
            "svm": {
                "cluster_id": svm_cluster,
                "category": svm_category,
                "confidence": round(class_probabilities[svm_cluster], 4),
            },
            "kmeans": {
                "cluster_id": kmeans_cluster,
                "category": kmeans_category,
            },
            "models_agree": svm_cluster == kmeans_cluster,
            "description": metadata["description"],
            "color": metadata["color"],
            "input_summary": {
                "age": employee.age,
                "years_experience": employee.years_experience,
                "education_level": employee.education_level,
                "education_label": EDUCATION_LEVELS[employee.education_level],
                "department": employee.department,
            },
            "process": {
                "preprocessing": build_preprocessing_trace(
                    self.bundle.preprocessor,
                    transformed,
                    [
                        employee.age,
                        employee.years_experience,
                        employee.education_level,
                    ],
                    employee.department,
                ),
                "kmeans": build_kmeans_trace(
                    self.bundle.kmeans,
                    transformed,
                    kmeans_cluster,
                    self.bundle.cluster_labels,
                ),
                "cluster_plot": {
                    **self.bundle.cluster_plot,
                    "new_point": project_new_point(
                        self.bundle.pca,
                        transformed,
                        kmeans_cluster,
                        self.bundle.cluster_labels,
                    ),
                },
                "svm": build_svm_trace(
                    class_probabilities,
                    svm_cluster,
                    self.bundle.cluster_labels,
                ),
                "svm_plot": {
                    **self.bundle.svm_plot,
                    "points": self.bundle.cluster_plot["points"],
                    "new_point": project_svm_new_point(
                        self.bundle.svm_visualizer,
                        projected_point,
                        svm_cluster,
                        self.bundle.cluster_labels,
                    ),
                },
            },
        }

    def model_info(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "dataset_rows": self.bundle.dataset_rows,
            "features": [
                "Age",
                "TotalWorkingYears",
                "Education",
                "Department",
            ],
            "categories": list(CATEGORY_METADATA),
            "cluster_profiles": list(self.bundle.cluster_profiles.values()),
            "cluster_plot": self.bundle.cluster_plot,
            "svm_plot": {
                **self.bundle.svm_plot,
                "points": self.bundle.cluster_plot["points"],
            },
            "metrics": {
                key: round(value, 4)
                for key, value in self.bundle.metrics.items()
            },
            "loaded_from_cache": self.loaded_from_cache,
            "pipeline_version": self.bundle.pipeline_version,
            "workflow": [
                "Input data",
                "StandardScaler + OneHotEncoder",
                "K-Means centroid distance",
                "RBF SVM classification",
            ],
        }
