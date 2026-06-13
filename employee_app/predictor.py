"""Validate employee input and produce predictions with an educational trace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from employee_app.config import (
    CATEGORY_METADATA,
    DEPARTMENTS,
    DISCLAIMER,
    EDUCATION_LEVELS,
    MODEL_FEATURES,
)
from employee_app.model_trainer import ModelBundle


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

    def as_model_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Age": self.age,
                    "TotalWorkingYears": self.years_experience,
                    "Education": self.education_level,
                    "Department": self.department,
                }
            ],
            columns=MODEL_FEATURES,
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
    def __init__(self, bundle: ModelBundle, loaded_from_cache: bool = False):
        self.bundle = bundle
        self.loaded_from_cache = loaded_from_cache

    def _preprocessing_trace(
        self,
        employee: EmployeeInput,
        transformed: np.ndarray,
    ) -> dict[str, Any]:
        """Explain how raw form values become the vector consumed by the models."""

        scaler = self.bundle.preprocessor.named_transformers_["numeric"]
        encoder = self.bundle.preprocessor.named_transformers_["department"]
        numeric_values = [
            employee.age,
            employee.years_experience,
            employee.education_level,
        ]

        numeric_trace = []
        for feature, value, mean, scale, standardized in zip(
            ["Age", "TotalWorkingYears", "Education"],
            numeric_values,
            scaler.mean_,
            scaler.scale_,
            transformed[0][:3],
            strict=True,
        ):
            numeric_trace.append(
                {
                    "feature": feature,
                    "original": value,
                    "dataset_mean": round(float(mean), 3),
                    "dataset_scale": round(float(scale), 3),
                    "standardized": round(float(standardized), 3),
                    "formula": (
                        f"({value} - {mean:.2f}) / {scale:.2f} "
                        f"= {standardized:.3f}"
                    ),
                }
            )

        department_categories = encoder.categories_[0].tolist()
        department_values = transformed[0][3:]
        one_hot = {
            department: int(encoded)
            for department, encoded in zip(
                department_categories,
                department_values,
                strict=True,
            )
        }

        feature_names = [
            feature_name.replace("numeric__", "").replace("department__", "")
            for feature_name in self.bundle.preprocessor.get_feature_names_out()
        ]
        return {
            "numeric_scaling": numeric_trace,
            "department_encoding": {
                "selected": employee.department,
                "one_hot": one_hot,
            },
            "feature_names": feature_names,
            "transformed_vector": [
                round(float(value), 3) for value in transformed[0]
            ],
        }

    def _kmeans_trace(
        self,
        transformed: np.ndarray,
        selected_cluster: int,
    ) -> dict[str, Any]:
        """Show Euclidean distance from the employee vector to every centroid."""

        distances = self.bundle.kmeans.transform(transformed)[0]
        ranked_distances = sorted(
            (
                {
                    "cluster_id": int(cluster_id),
                    "category": self.bundle.cluster_labels[int(cluster_id)],
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
            "nearest_category": self.bundle.cluster_labels[selected_cluster],
        }

    def _svm_trace(
        self,
        class_probabilities: dict[int, float],
        selected_cluster: int,
    ) -> dict[str, Any]:
        """Show calibrated SVM probabilities for each learned cluster label."""

        probabilities = sorted(
            (
                {
                    "cluster_id": cluster_id,
                    "category": self.bundle.cluster_labels[cluster_id],
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
            "selected_category": self.bundle.cluster_labels[selected_cluster],
        }

    def predict(self, payload: Any) -> dict[str, Any]:
        """Validate one employee, run both models, and return an explainable trace."""

        employee = validate_employee_input(payload)
        transformed = self.bundle.preprocessor.transform(employee.as_model_frame())

        kmeans_cluster = int(self.bundle.kmeans.predict(transformed)[0])
        svm_cluster = int(self.bundle.svm.predict(transformed)[0])
        probabilities = self.bundle.svm.predict_proba(transformed)[0]
        class_probabilities = {
            int(class_id): float(probability)
            for class_id, probability in zip(
                self.bundle.svm.classes_,
                probabilities,
                strict=True,
            )
        }

        svm_category = self.bundle.cluster_labels[svm_cluster]
        kmeans_category = self.bundle.cluster_labels[kmeans_cluster]
        metadata = CATEGORY_METADATA[svm_category]

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
                "input_validation": {
                    "status": "valid",
                    "message": (
                        "Semua nilai berada dalam rentang dataset dan siap "
                        "diproses."
                    ),
                },
                "preprocessing": self._preprocessing_trace(
                    employee,
                    transformed,
                ),
                "kmeans": self._kmeans_trace(
                    transformed,
                    kmeans_cluster,
                ),
                "svm": self._svm_trace(
                    class_probabilities,
                    svm_cluster,
                ),
            },
            "disclaimer": DISCLAIMER,
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
            "metrics": {
                key: round(value, 4)
                for key, value in self.bundle.metrics.items()
            },
            "loaded_from_cache": self.loaded_from_cache,
            "pipeline_version": self.bundle.pipeline_version,
            "workflow": [
                "Validasi input",
                "StandardScaler dan OneHotEncoder",
                "Perhitungan jarak centroid K-Means",
                "Klasifikasi dan probabilitas SVM",
            ],
            "disclaimer": DISCLAIMER,
        }
