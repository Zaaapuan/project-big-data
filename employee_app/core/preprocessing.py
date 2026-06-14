"""Convert raw employee data into the numeric vector used by both models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from employee_app.core.config import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
)


def build_preprocessor() -> ColumnTransformer:
    """Build the shared StandardScaler and OneHotEncoder pipeline."""

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


def employee_frame(
    age: int,
    years_experience: int,
    education_level: int,
    department: str,
) -> pd.DataFrame:
    """Map form field names to the column names used by the dataset."""

    return pd.DataFrame(
        [
            {
                "Age": age,
                "TotalWorkingYears": years_experience,
                "Education": education_level,
                "Department": department,
            }
        ],
        columns=MODEL_FEATURES,
    )


def build_preprocessing_trace(
    preprocessor: ColumnTransformer,
    transformed: np.ndarray,
    numeric_values: list[int],
    department: str,
) -> dict[str, Any]:
    """Explain standardization and one-hot encoding using fitted values."""

    scaler = preprocessor.named_transformers_["numeric"]
    encoder = preprocessor.named_transformers_["department"]

    numeric_trace = []
    for feature, value, mean, scale, standardized in zip(
        NUMERIC_FEATURES,
        numeric_values,
        scaler.mean_,
        scaler.scale_,
        transformed[0][: len(NUMERIC_FEATURES)],
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

    category_values = transformed[0][len(NUMERIC_FEATURES) :]
    one_hot = {
        category: int(encoded)
        for category, encoded in zip(
            encoder.categories_[0].tolist(),
            category_values,
            strict=True,
        )
    }
    feature_names = [
        name.replace("numeric__", "").replace("department__", "")
        for name in preprocessor.get_feature_names_out()
    ]

    return {
        "numeric_scaling": numeric_trace,
        "department_encoding": {
            "selected": department,
            "one_hot": one_hot,
        },
        "feature_names": feature_names,
        "transformed_vector": [
            round(float(value), 3) for value in transformed[0]
        ],
    }
