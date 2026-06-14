"""Load and validate the local IBM HR dataset before model training."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from employee_app.core.config import DEPARTMENTS, MODEL_FEATURES


class DatasetError(ValueError):
    """Raised when the local dataset cannot be used by the model."""


def dataset_hash(path: Path) -> str:
    """Return a SHA-256 hash used to detect dataset changes."""

    digest = hashlib.sha256()
    with path.open("rb") as dataset_file:
        for chunk in iter(lambda: dataset_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path: Path) -> pd.DataFrame:
    """Read the four model features and validate their expected values."""

    if not path.exists():
        raise DatasetError(f"Dataset lokal tidak ditemukan: {path}")

    dataframe = pd.read_csv(path, encoding="utf-8-sig")
    missing_columns = sorted(set(MODEL_FEATURES) - set(dataframe.columns))
    if missing_columns:
        raise DatasetError(
            "Dataset tidak memiliki kolom wajib: " + ", ".join(missing_columns)
        )

    model_data = dataframe[MODEL_FEATURES].copy()
    if model_data.isna().any().any():
        raise DatasetError("Dataset memiliki nilai kosong pada fitur model.")

    for column in ("Age", "TotalWorkingYears", "Education"):
        model_data[column] = pd.to_numeric(model_data[column], errors="raise")

    invalid_departments = sorted(
        set(model_data["Department"].unique()) - set(DEPARTMENTS)
    )
    if invalid_departments:
        raise DatasetError(
            "Dataset memiliki departemen tidak dikenal: "
            + ", ".join(invalid_departments)
        )

    if not model_data["Age"].between(18, 60).all():
        raise DatasetError("Nilai Age harus berada dalam rentang 18-60.")
    if not model_data["TotalWorkingYears"].between(0, 40).all():
        raise DatasetError(
            "Nilai TotalWorkingYears harus berada dalam rentang 0-40."
        )
    if not model_data["Education"].isin(range(1, 6)).all():
        raise DatasetError("Nilai Education harus berupa kode 1-5.")

    return model_data
